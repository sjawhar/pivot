"""Watch-mode code reload through a running Engine, driven by real pipeline.py edits.

Each test writes a project, discovers its pipeline the way the CLI does, runs an Engine with a
FilesystemSource on the project root (the wiring ``pivot repro --watch`` uses), edits pipeline.py
on disk and observes the engine's output events and the files its stages write.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING

import anyio
import pytest
import watchfiles

from pivot import cli, discovery, exceptions, project
from pivot.engine import agent_rpc, sources
from pivot.engine import engine as engine_mod
from pivot.engine.types import EngineState, InputEvent, OutputEvent
from pivot.storage import store as store_mod
from pivot.storage import track
from pivot.types import StageStatus
from pivot_tui import event_poller, rpc_client_impl

if TYPE_CHECKING:
    import pathlib
    from collections.abc import AsyncGenerator, Callable

    import click.testing
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

    from pivot.registry import PipelineLike


def _helper_pipeline_source(change_suffix: str, *, third: str, refused: bool = False) -> str:
    """pipeline.py with stages keep, change (input + change_suffix) and ``third``.

    With ``refused``, ``change`` reads its suffix from a module-level list, which
    fingerprinting refuses.
    """
    suffixes = f"SUFFIXES = [{change_suffix!r}]\n\n\n" if refused else ""
    change_value = "SUFFIXES[0]" if refused else repr(change_suffix)
    return f"""\
from __future__ import annotations

from pivot.compose import Pipeline, stage

{suffixes}
@stage
def keep(raw: str) -> str:
    return raw + "-keep"


@stage
def change(raw: str) -> str:
    return raw + {change_value}


@stage
def {third}(raw: str) -> str:
    return raw + "-{third}"


with Pipeline("reload") as pipeline:
    raw = pipeline.input("input.txt")
    keep(raw)
    change(raw)
    {third}(raw)
"""


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

    def completed(self, stage: str, *, start: int, end: int | None = None) -> StageStatus:
        """Status of the first completion of ``stage`` in events[start:end]."""
        for event in self.events[start:end]:
            if event["type"] == "stage_completed" and event["stage"] == stage:
                return event["status"]
        raise AssertionError(f"{stage} did not complete in events[{start}:{end}]")


type _Batch = set[tuple[watchfiles.Change, str]]


class _ScriptedWatch:
    """Stands in for ``watchfiles.awatch`` and yields exactly the batches a test reports.

    With real watching, the OS and the debounce decide which changes share a batch; here the
    test does. The watched paths do not matter, so a watch-path update does not restart it.
    """

    _send: MemoryObjectSendStream[_Batch]
    _recv: MemoryObjectReceiveStream[_Batch]

    def __init__(self) -> None:
        self._send, self._recv = anyio.create_memory_object_stream[_Batch](8)

    async def report(self, *paths: pathlib.Path) -> None:
        """Report ``paths`` as modified, all in one batch."""
        await self._send.send({(watchfiles.Change.modified, str(path)) for path in paths})

    async def awatch(
        self, *_paths: object, debounce: int, stop_event: anyio.Event
    ) -> AsyncGenerator[_Batch]:
        async for batch in self._recv:
            yield batch


def _helper_is_idle(event: OutputEvent) -> bool:
    return event["type"] == "engine_state_changed" and event["state"] == EngineState.IDLE


def _helper_is_reload(event: OutputEvent) -> bool:
    """A reload that loaded a new pipeline."""
    return event["type"] == "pipeline_reloaded" and event["error"] is None


def _helper_is_invalid_pipeline(event: OutputEvent) -> bool:
    """A reload or startup snapshot that found the pipeline invalid."""
    return event["type"] == "pipeline_reloaded" and event["error"] is not None


def _helper_completes(stage: str) -> Callable[[OutputEvent], bool]:
    def _predicate(event: OutputEvent) -> bool:
        return event["type"] == "stage_completed" and event["stage"] == stage

    return _predicate


def _helper_output_path(root: pathlib.Path, pipeline: PipelineLike, stage: str) -> pathlib.Path:
    store = store_mod.WorkspaceStore(
        project_root=root,
        pipeline_name=pipeline.name,
        input_bindings=pipeline.input_bindings,
    )
    return store.resolve_display_path(pipeline.get_stage(stage)["outs"][0])


@pytest.fixture
def watch_project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    root = tmp_path.resolve()
    (root / ".pivot").mkdir()
    (root / ".git").mkdir()
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "input.txt").write_text("x")
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", root)
    return root


def _helper_discover(root: pathlib.Path) -> PipelineLike:
    pipeline = discovery.discover_pipeline(root)
    assert pipeline is not None, "pipeline.py should define a pipeline"
    return pipeline


async def _helper_wait_until_watching(eng: engine_mod.Engine) -> None:
    """Return once the engine's startup snapshot is done and its file watcher is running."""
    await eng.wait_for_fingerprints()
    await anyio.wait_all_tasks_blocked()


@pytest.mark.anyio
async def test_pipeline_edit_reloads_and_reruns_changed_stages_with_new_code(
    watch_project: pathlib.Path,
) -> None:
    """Editing pipeline.py reports the stage diff, then reruns edited and added stages."""
    pipeline_py = watch_project / "pipeline.py"
    pipeline_py.write_text(_helper_pipeline_source("-v1", third="removed"))
    pipeline = _helper_discover(watch_project)
    change_output = _helper_output_path(watch_project, pipeline, "reload/change")
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        eng.add_source(
            sources.OneShotSource(stages=None, force=False, reason="test", max_workers=1)
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))

            first_idle = await recorder.wait_for(_helper_is_idle)
            assert change_output.read_text() == "x-v1", "initial run writes v1 output"

            pipeline_py.write_text(_helper_pipeline_source("-version2", third="added"))
            reload_index = await recorder.wait_for(_helper_is_reload, start=first_idle)
            last_completion = max(
                [
                    await recorder.wait_for(_helper_completes(stage), start=reload_index)
                    for stage in ("reload/keep", "reload/change", "reload/added")
                ]
            )
            await recorder.wait_for(_helper_is_idle, start=last_completion)

            reload_event = recorder.events[reload_index]
            assert reload_event["type"] == "pipeline_reloaded"
            assert sorted(reload_event["stages"]) == [
                "reload/added",
                "reload/change",
                "reload/keep",
            ]
            assert reload_event["stages_added"] == ["reload/added"]
            assert reload_event["stages_removed"] == ["reload/removed"]
            assert reload_event["stages_modified"] == ["reload/change"], (
                "only the stage whose body changed is modified"
            )
            assert reload_event["error"] is None

            for stage in ("reload/keep", "reload/change", "reload/removed"):
                assert recorder.completed(stage, start=0, end=first_idle) == StageStatus.RAN
            assert recorder.completed("reload/change", start=reload_index) == StageStatus.RAN
            assert recorder.completed("reload/added", start=reload_index) == StageStatus.RAN
            assert recorder.completed("reload/keep", start=reload_index) == StageStatus.CACHED, (
                "an unchanged stage stays cached across the reload"
            )
            assert change_output.read_text() == "x-version2", "workers run the reloaded stage code"
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_pipeline_edit_before_any_run_reports_edited_stage_as_modified(
    watch_project: pathlib.Path,
) -> None:
    """Without an initial run (``repro --watch`` without --force), an edit is still a diff."""
    pipeline_py = watch_project / "pipeline.py"
    pipeline_py.write_text(_helper_pipeline_source("-v1", third="other"))
    pipeline = _helper_discover(watch_project)
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            await _helper_wait_until_watching(eng)

            pipeline_py.write_text(_helper_pipeline_source("-v2", third="other"))
            reload_index = await recorder.wait_for(_helper_is_reload)
            await recorder.wait_for(_helper_is_idle, start=reload_index)

            reload_event = recorder.events[reload_index]
            assert reload_event["type"] == "pipeline_reloaded"
            assert reload_event["stages_added"] == []
            assert reload_event["stages_removed"] == []
            assert reload_event["stages_modified"] == ["reload/change"], (
                "the edited stage is modified even though no run fingerprinted it before the edit"
            )
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_same_size_edit_within_one_second_reloads_new_code(
    watch_project: pathlib.Path,
) -> None:
    """An edit that keeps pipeline.py's size and whole-second mtime still reloads its new code.

    CPython validates cached bytecode by source size and mtime in whole seconds, so such an
    edit would otherwise import the previous ``__pycache__`` entry.
    """
    pipeline_py = watch_project / "pipeline.py"
    pipeline_py.write_text(_helper_pipeline_source("-v1", third="removed"))
    pipeline = _helper_discover(watch_project)
    first = pipeline_py.stat()
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            await _helper_wait_until_watching(eng)

            edited = _helper_pipeline_source("-v1", third="renamed")
            assert len(edited) == first.st_size
            pipeline_py.write_text(edited)
            os.utime(pipeline_py, ns=(first.st_atime_ns, first.st_mtime_ns))
            reload_index = await recorder.wait_for(_helper_is_reload)
            reload_event = recorder.events[reload_index]
            assert reload_event["type"] == "pipeline_reloaded"
            assert reload_event["stages_added"] == ["reload/renamed"], (
                "the reload must import the edited source, not stale bytecode"
            )
            assert reload_event["stages_removed"] == ["reload/removed"]
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_edit_with_read_only_pycache_still_reloads_new_code(
    watch_project: pathlib.Path,
) -> None:
    """A same-size edit within one second reloads its new code even when the bytecode cached
    for the old code cannot be replaced, because ``__pycache__`` is read-only."""
    pipeline_py = watch_project / "pipeline.py"
    pipeline_py.write_text(_helper_pipeline_source("-v1", third="removed"))
    pipeline = _helper_discover(watch_project)
    pycache = watch_project / "__pycache__"
    assert any(pycache.glob("pipeline.*.pyc")), "discovery caches pipeline.py's bytecode"
    first = pipeline_py.stat()
    recorder = _Recorder()

    pycache.chmod(0o555)
    try:
        async with engine_mod.Engine(pipeline=pipeline) as eng:
            eng.add_sink(recorder)
            eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
            async with anyio.create_task_group() as tg:
                tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
                await _helper_wait_until_watching(eng)

                edited = _helper_pipeline_source("-v1", third="renamed")
                assert len(edited) == first.st_size
                pipeline_py.write_text(edited)
                os.utime(pipeline_py, ns=(first.st_atime_ns, first.st_mtime_ns))
                reload_index = await recorder.wait_for(_helper_is_reload)
                reload_event = recorder.events[reload_index]
                assert reload_event["type"] == "pipeline_reloaded"
                assert reload_event["stages_added"] == ["reload/renamed"]
                tg.cancel_scope.cancel()
    finally:
        pycache.chmod(0o755)


@pytest.mark.anyio
async def test_watched_startup_fingerprints_each_stage_once(
    watch_project: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-start fingerprint snapshot and the startup run's skip check share one computation.

    The snapshot finishes before the file watcher starts or any input event is handled, so the
    run reuses its fingerprints and no stage is fingerprinted on two threads at once.
    """
    from pivot import registry

    (watch_project / "pipeline.py").write_text(_helper_pipeline_source("-v1", third="other"))
    pipeline = _helper_discover(watch_project)
    calls = list[str]()
    compute = registry._compute_fingerprint

    def _counting_compute(
        stage_name: str, info: registry.RegistryStageInfo
    ) -> tuple[dict[str, str], bool]:
        calls.append(stage_name)
        return compute(stage_name, info)

    monkeypatch.setattr(registry, "_compute_fingerprint", _counting_compute)
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        eng.add_source(
            sources.OneShotSource(stages=None, force=False, reason="test", max_workers=1)
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            await recorder.wait_for(_helper_is_idle)
            tg.cancel_scope.cancel()

    for stage in ("reload/keep", "reload/change", "reload/other"):
        assert recorder.completed(stage, start=0) == StageStatus.RAN
    assert sorted(calls) == ["reload/change", "reload/keep", "reload/other"], (
        "each stage is fingerprinted exactly once per startup"
    )


@pytest.mark.anyio
async def test_watched_startup_serves_rpc_while_fingerprinting(
    watch_project: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RPC socket listens before a slow startup fingerprint snapshot finishes.

    ``status`` answers while the snapshot runs; ``explain`` waits for it, so the snapshot stays
    the only code fingerprinting and each stage is still fingerprinted once.
    """
    from pivot import registry

    (watch_project / "pipeline.py").write_text(_helper_pipeline_source("-v1", third="other"))
    pipeline = _helper_discover(watch_project)
    compute = registry._compute_fingerprint
    calls = list[str]()
    finished = list[str]()

    def _slow_compute(
        stage_name: str, info: registry.RegistryStageInfo
    ) -> tuple[dict[str, str], bool]:
        calls.append(stage_name)
        time.sleep(1.0)
        result = compute(stage_name, info)
        finished.append(stage_name)
        return result

    monkeypatch.setattr(registry, "_compute_fingerprint", _slow_compute)
    recorder = _Recorder()
    socket_path = watch_project / "rpc.sock"
    ready = threading.Event()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        handler = agent_rpc.AgentRpcHandler(engine=eng)
        eng.add_source(
            agent_rpc.AgentRpcSource(socket_path=socket_path, handler=handler, ready=ready)
        )
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        eng.add_source(
            sources.OneShotSource(stages=None, force=False, reason="test", max_workers=1)
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            client = rpc_client_impl.RpcPivotClient()
            with anyio.fail_after(2):
                while not ready.is_set():
                    await anyio.sleep(0.01)
                await client.connect(socket_path)
            try:
                status = await client.status()
                assert status["state"] == "idle"
                assert len(finished) < 3, "status is answered while the snapshot still runs"

                explanation = await client.explain("reload/change")
                assert explanation["stage_name"] == "reload/change"
                assert sorted(finished) == ["reload/change", "reload/keep", "reload/other"], (
                    "explain answers only after the snapshot has fingerprinted every stage"
                )
            finally:
                await client.disconnect()
            await recorder.wait_for(_helper_is_idle)
            tg.cancel_scope.cancel()

    assert sorted(calls) == ["reload/change", "reload/keep", "reload/other"], (
        "each stage is fingerprinted exactly once, even with explain issued mid-snapshot"
    )


@pytest.mark.anyio
async def test_reload_fingerprints_off_the_event_loop(
    watch_project: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While a reload imports and fingerprints slowly, the RPC server keeps answering.

    ``status`` answers within 1 s mid-reload, and ``run`` and ``commit`` are refused at once
    with -32004 "Engine is reloading" until the new pipeline is in.
    """
    from pivot import registry

    pipeline_py = watch_project / "pipeline.py"
    pipeline_py.write_text(_helper_pipeline_source("-v1", third="removed"))
    pipeline = _helper_discover(watch_project)
    recorder = _Recorder()
    socket_path = watch_project / "rpc.sock"
    ready = threading.Event()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        handler = agent_rpc.AgentRpcHandler(engine=eng)
        eng.add_source(
            agent_rpc.AgentRpcSource(socket_path=socket_path, handler=handler, ready=ready)
        )
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            client = rpc_client_impl.RpcPivotClient()
            with anyio.fail_after(5):
                while not ready.is_set():
                    await anyio.sleep(0.01)
                await client.connect(socket_path)
            await _helper_wait_until_watching(eng)

            compute = registry._compute_fingerprint
            reload_started = threading.Event()
            finished = list[str]()

            def _slow_compute(
                stage_name: str, info: registry.RegistryStageInfo
            ) -> tuple[dict[str, str], bool]:
                reload_started.set()
                time.sleep(1.0)
                finished.append(stage_name)
                return compute(stage_name, info)

            monkeypatch.setattr(registry, "_compute_fingerprint", _slow_compute)
            pipeline_py.write_text(_helper_pipeline_source("-v1", third="added"))
            try:
                with anyio.fail_after(10):
                    while not reload_started.is_set():
                        await anyio.sleep(0.01)

                start = time.monotonic()
                with anyio.fail_after(5):
                    status = await client.status()
                assert time.monotonic() - start < 1.0, "status answers within 1 s mid-reload"
                assert status["state"] == "idle"
                assert len(finished) < 3, "status answered while the reload still fingerprints"

                for request in (client.run(force=True), client.commit()):
                    with anyio.fail_after(1), pytest.raises(rpc_client_impl.RpcError) as refused:
                        await request
                    assert refused.value.code == -32004
                    assert "Engine is reloading" in refused.value.message
                assert len(finished) < 3, "run and commit were refused before the reload ended"
            finally:
                await client.disconnect()

            reload_index = await recorder.wait_for(_helper_is_reload)
            reload_event = recorder.events[reload_index]
            assert reload_event["type"] == "pipeline_reloaded"
            assert reload_event["stages_added"] == ["reload/added"]
            assert reload_event["stages_removed"] == ["reload/removed"]
            done = await recorder.wait_for(_helper_completes("reload/added"), start=reload_index)
            await recorder.wait_for(_helper_is_idle, start=done)
            assert not eng.is_reloading, "the reload ends once its re-run has started"
            tg.cancel_scope.cancel()


def _helper_chains_source(b_suffix: str, *, a_sleep: float = 0.0) -> str:
    """pipeline.py with two independent stages: from_a reads a.txt and from_b reads b.txt.

    With ``a_sleep``, from_a sleeps that many seconds, so a run stays busy in it.
    """
    return f"""\
from __future__ import annotations

import time

from pivot.compose import Pipeline, stage


@stage
def from_a(raw: str) -> str:
    time.sleep({a_sleep!r})
    return raw + "-a"


@stage
def from_b(raw: str) -> str:
    return raw + {b_suffix!r}


with Pipeline("chains") as pipeline:
    from_a(pipeline.input("a.txt"))
    from_b(pipeline.input("b.txt"))
"""


def _helper_nested_input_source() -> str:
    """pipeline.py with a directory input and a file inside that directory."""
    return """\
import pathlib

from pivot import Pipeline, stage


@stage
def directory(folder: pathlib.Path) -> str:
    return (folder / "a.txt").read_text()


@stage
def file(data: str) -> str:
    return data


with Pipeline("nested") as pipeline:
    directory(pipeline.input("folder", path="inputs", t=pathlib.Path))
    file(pipeline.input("file", path="inputs/a.txt", t=str))
"""


@pytest.mark.anyio
async def test_child_input_edit_reruns_directory_and_file_consumers(
    watch_project: pathlib.Path,
) -> None:
    """Changing a child input runs consumers of both the file and its directory."""
    inputs = watch_project / "inputs"
    inputs.mkdir()
    child = inputs / "a.txt"
    child.write_text("one")
    (watch_project / "pipeline.py").write_text(_helper_nested_input_source())
    pipeline = _helper_discover(watch_project)
    directory_output = _helper_output_path(watch_project, pipeline, "nested/directory")
    file_output = _helper_output_path(watch_project, pipeline, "nested/file")
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        eng.add_source(
            sources.OneShotSource(stages=None, force=False, reason="test", max_workers=2)
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            first_idle = await recorder.wait_for(_helper_is_idle)
            assert (directory_output.read_text(), file_output.read_text()) == ("one", "one")

            child.write_text("two")
            directory_done = await recorder.wait_for(
                _helper_completes("nested/directory"), start=first_idle, timeout=10
            )
            file_done = await recorder.wait_for(
                _helper_completes("nested/file"), start=first_idle, timeout=10
            )
            await recorder.wait_for(
                _helper_is_idle, start=max(directory_done, file_done), timeout=10
            )
            assert (directory_output.read_text(), file_output.read_text()) == ("two", "two")
            tg.cancel_scope.cancel()


def _helper_symlinked_input_source() -> str:
    """pipeline.py with direct, directory, and symlink aliases for one input file."""
    return """\
import pathlib

from pivot import Pipeline, stage


@stage
def whole(folder: pathlib.Path) -> str:
    return (folder / "sub/a.txt").read_text()


@stage
def nested(folder: pathlib.Path) -> str:
    return (folder / "a.txt").read_text()


@stage
def direct(data: str) -> str:
    return data


@stage
def aliasdir(folder: pathlib.Path) -> str:
    return (folder / "a.txt").read_text()


@stage
def aliasfile(data: str) -> str:
    return data


@stage
def unrelated(data: str) -> str:
    return data


with Pipeline("aliases") as pipeline:
    whole(pipeline.input("whole", path="inputs", t=pathlib.Path))
    nested(pipeline.input("nested", path="inputs/sub", t=pathlib.Path))
    direct(pipeline.input("direct", path="inputs/sub/a.txt", t=str))
    aliasdir(pipeline.input("aliasdir", path="alias", t=pathlib.Path))
    aliasfile(pipeline.input("aliasfile", path="alias-file.txt", t=str))
    unrelated(pipeline.input("unrelated", path="other/a.txt", t=str))
"""


@pytest.mark.anyio
async def test_input_edits_reach_consumers_through_symlinked_paths(
    watch_project: pathlib.Path,
) -> None:
    """Physical and symlinked edits rerun every consumer of the same input file."""
    inputs = watch_project / "inputs"
    child = inputs / "sub" / "a.txt"
    child.parent.mkdir(parents=True)
    child.write_text("one")
    other = watch_project / "other" / "a.txt"
    other.parent.mkdir()
    other.write_text("other")
    (watch_project / "alias").symlink_to(inputs / "sub", target_is_directory=True)
    (watch_project / "alias-file.txt").symlink_to(child)
    (watch_project / "pipeline.py").write_text(_helper_symlinked_input_source())
    pipeline = _helper_discover(watch_project)
    affected_stages = {
        "aliases/whole",
        "aliases/nested",
        "aliases/direct",
        "aliases/aliasdir",
        "aliases/aliasfile",
    }
    outputs = {
        stage: _helper_output_path(watch_project, pipeline, stage)
        for stage in affected_stages | {"aliases/unrelated"}
    }
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        eng.add_source(
            sources.OneShotSource(stages=None, force=False, reason="test", max_workers=6)
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            await recorder.wait_for(_helper_is_idle)
            assert {outputs[stage].read_text() for stage in affected_stages} == {"one"}
            assert outputs["aliases/unrelated"].read_text() == "other"

            for event_path, expected in (
                (child, "two"),
                (watch_project / "alias" / "a.txt", "three"),
            ):
                start = len(recorder.events)
                event_path.write_text(expected)
                completed = set[str]()
                for stage in affected_stages:
                    done = await recorder.wait_for(
                        _helper_completes(stage), start=start, timeout=10
                    )
                    event = recorder.events[done]
                    assert event["type"] == "stage_completed"
                    completed.add(event["stage"])
                idle = await recorder.wait_for(_helper_is_idle, start=start, timeout=10)
                for event in recorder.events[start:idle]:
                    if event["type"] == "stage_completed":
                        completed.add(event["stage"])
                assert completed == affected_stages
                assert {outputs[stage].read_text() for stage in affected_stages} == {expected}
                assert outputs["aliases/unrelated"].read_text() == "other"
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_data_edit_during_reload_still_reruns_the_code_edited_stage(
    watch_project: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A data edit made while a slow reload runs does not narrow the reload's re-run.

    The engine handles the a.txt edit once the re-run of every stage has started, and the restart
    it causes must still re-run from_b, whose code the reload changed, as well as from_a.
    """
    from pivot import registry

    raw_dir = watch_project / "data" / "raw"
    (raw_dir / "a.txt").write_text("a1")
    (raw_dir / "b.txt").write_text("b1")
    pipeline_py = watch_project / "pipeline.py"
    pipeline_py.write_text(_helper_chains_source("-v1"))
    pipeline = _helper_discover(watch_project)
    a_output = _helper_output_path(watch_project, pipeline, "chains/from_a")
    b_output = _helper_output_path(watch_project, pipeline, "chains/from_b")
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        # One worker: from_a runs first, so from_b is still queued when the restart cancels
        eng.add_source(
            sources.OneShotSource(stages=None, force=False, reason="test", max_workers=1)
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            first_idle = await recorder.wait_for(_helper_is_idle)
            assert (a_output.read_text(), b_output.read_text()) == ("a1-a", "b1-v1")

            compute = registry._compute_fingerprint
            reload_started = threading.Event()

            def _slow_compute(
                stage_name: str, info: registry.RegistryStageInfo
            ) -> tuple[dict[str, str], bool]:
                reload_started.set()
                time.sleep(1.0)
                return compute(stage_name, info)

            monkeypatch.setattr(registry, "_compute_fingerprint", _slow_compute)
            pipeline_py.write_text(_helper_chains_source("-v2"))
            with anyio.fail_after(10):
                while not reload_started.is_set():
                    await anyio.sleep(0.01)
            (raw_dir / "a.txt").write_text("a2")

            reload_index = await recorder.wait_for(_helper_is_reload, start=first_idle)
            await recorder.wait_for(_helper_is_idle, start=reload_index)
            assert a_output.read_text() == "a2-a", "the mid-reload data edit is picked up"
            assert b_output.read_text() == "b1-v2", "the code-edited stage re-runs"
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_data_edit_while_a_run_is_cancelling_waits_for_it(
    watch_project: pathlib.Path,
) -> None:
    """A data edit made while a run is cancelling keeps watch alive and is picked up afterwards.

    Editing a.txt twice cancels the first from_a run, which stays CANCELLING until its stage
    ends. A b.txt edit then joins the pending restart instead of starting a second run against
    a worker pool that has stopped accepting stages.
    """
    raw_dir = watch_project / "data" / "raw"
    (raw_dir / "a.txt").write_text("a1")
    (raw_dir / "b.txt").write_text("b1")
    (watch_project / "pipeline.py").write_text(_helper_chains_source("-b", a_sleep=1.5))
    pipeline = _helper_discover(watch_project)
    a_output = _helper_output_path(watch_project, pipeline, "chains/from_a")
    b_output = _helper_output_path(watch_project, pipeline, "chains/from_b")
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        eng.add_source(sources.OneShotSource(stages=None, force=False, reason="test"))
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            first_idle = await recorder.wait_for(_helper_is_idle)

            (raw_dir / "a.txt").write_text("a2")
            await recorder.wait_for(
                lambda e: e["type"] == "stage_started" and e["stage"] == "chains/from_a",
                start=first_idle,
            )
            (raw_dir / "a.txt").write_text("a3")
            with anyio.fail_after(5):
                while eng._run_state != engine_mod._RunState.CANCELLING:
                    await anyio.sleep(0.01)
            b_edit = len(recorder.events)
            (raw_dir / "b.txt").write_text("b2")

            done = await recorder.wait_for(_helper_completes("chains/from_b"), start=b_edit)
            await recorder.wait_for(_helper_is_idle, start=done)
            assert a_output.read_text() == "a3-a", "the second a.txt edit is picked up"
            assert b_output.read_text() == "b2-b", "the b.txt edit made while cancelling is too"
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_pipeline_and_data_edits_in_one_batch_run_each_stage_once(
    watch_project: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pipeline.py edit the watcher reports in one batch with a data edit runs each stage once.

    The data edit was made before the reload the batch starts, so the reload's re-run covers
    it, and the re-run is not cancelled and restarted for it. An editor that saves every buffer
    at once, or a checkout that changes both files, makes such a batch.
    """
    watch = _ScriptedWatch()
    monkeypatch.setattr(watchfiles, "awatch", watch.awatch)
    pipeline_py = watch_project / "pipeline.py"
    input_txt = watch_project / "data" / "raw" / "input.txt"
    pipeline_py.write_text(_helper_pipeline_source("-v1", third="removed"))
    pipeline = _helper_discover(watch_project)
    keep_output = _helper_output_path(watch_project, pipeline, "reload/keep")
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            await _helper_wait_until_watching(eng)

            pipeline_py.write_text(_helper_pipeline_source("-v1", third="added"))
            input_txt.write_text("y")
            await watch.report(pipeline_py, input_txt)
            reload_index = await recorder.wait_for(_helper_is_reload)
            await recorder.wait_for(_helper_is_idle, start=reload_index)

            completions = sorted(
                (event["stage"], event["status"])
                for event in recorder.events[reload_index:]
                if event["type"] == "stage_completed"
            )
            assert completions == [
                ("reload/added", StageStatus.RAN),
                ("reload/change", StageStatus.RAN),
                ("reload/keep", StageStatus.RAN),
            ], "each stage runs once: the batch's data edit does not cancel the reload's re-run"
            assert keep_output.read_text() == "y-keep", "the re-run reads the data edit"
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_still_broken_pipeline_and_data_edit_in_one_batch_run_nothing_until_fixed(
    watch_project: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A batch with a still-broken pipeline.py and a data edit handles the data edit as an
    invalid pipeline does: it is dropped, and the edit that fixes the pipeline reruns it.

    The reload this batch starts leaves the pipeline invalid and starts no re-run, so nothing
    covers the data edit but the data-change handler.
    """
    caplog.set_level(logging.DEBUG, logger="pivot.engine.engine")
    watch = _ScriptedWatch()
    monkeypatch.setattr(watchfiles, "awatch", watch.awatch)
    pipeline_py = watch_project / "pipeline.py"
    input_txt = watch_project / "data" / "raw" / "input.txt"
    pipeline_py.write_text(_helper_pipeline_source("-v1", third="removed"))
    pipeline = _helper_discover(watch_project)
    keep_output = _helper_output_path(watch_project, pipeline, "reload/keep")
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            await _helper_wait_until_watching(eng)

            pipeline_py.write_text("def broken(:\n")
            await watch.report(pipeline_py)
            first_invalid = await recorder.wait_for(_helper_is_invalid_pipeline)

            input_txt.write_text("y")
            await watch.report(pipeline_py, input_txt)
            await recorder.wait_for(_helper_is_invalid_pipeline, start=first_invalid + 1)
            with anyio.fail_after(10):
                while "ignoring run request" not in caplog.text:
                    await anyio.sleep(0.05)
            assert "re-run covers the change" not in caplog.text, "no re-run started to cover it"

            pipeline_py.write_text(_helper_pipeline_source("-v1", third="added"))
            await watch.report(pipeline_py)
            reload_index = await recorder.wait_for(_helper_is_reload)
            await recorder.wait_for(_helper_is_idle, start=reload_index)

            assert not any(
                event["type"] == "stage_completed" for event in recorder.events[:reload_index]
            ), "nothing ran while the pipeline was invalid"
            assert keep_output.read_text() == "y-keep", "the fixed pipeline's run reads the edit"
            tg.cancel_scope.cancel()


@pytest.mark.anyio
@pytest.mark.parametrize("refused", [False, True], ids=["valid", "refused"])
async def test_run_requested_during_startup_snapshot_is_refused_at_once(
    watch_project: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    refused: bool,
) -> None:
    """A run sent while the startup snapshot runs gets -32004 at once and queues nothing.

    After the snapshot the usual rules apply: -32003 for a refused pipeline, accepted otherwise.
    """
    from pivot import registry

    (watch_project / "pipeline.py").write_text(
        _helper_pipeline_source("-v1", third="other", refused=refused)
    )
    pipeline = _helper_discover(watch_project)
    compute = registry._compute_fingerprint
    finished = list[str]()

    def _slow_compute(
        stage_name: str, info: registry.RegistryStageInfo
    ) -> tuple[dict[str, str], bool]:
        time.sleep(1.0)
        finished.append(stage_name)
        return compute(stage_name, info)

    monkeypatch.setattr(registry, "_compute_fingerprint", _slow_compute)
    recorder = _Recorder()
    socket_path = watch_project / "rpc.sock"
    ready = threading.Event()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        handler = agent_rpc.AgentRpcHandler(engine=eng)
        eng.add_source(
            agent_rpc.AgentRpcSource(socket_path=socket_path, handler=handler, ready=ready)
        )
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            client = rpc_client_impl.RpcPivotClient()
            with anyio.fail_after(2):
                while not ready.is_set():
                    await anyio.sleep(0.01)
                await client.connect(socket_path)
            try:
                with anyio.fail_after(0.5), pytest.raises(rpc_client_impl.RpcError) as starting:
                    await client.run(stages=["reload/change"], force=True)
                assert starting.value.code == -32004
                assert "Engine is starting" in starting.value.message
                assert len(finished) < 3, "the reply came before the snapshot finished"

                await _helper_wait_until_watching(eng)
                if refused:
                    with pytest.raises(rpc_client_impl.RpcError) as invalid:
                        await client.run(stages=["reload/change"], force=True)
                    assert invalid.value.code == -32003
                    assert "captures non-scalar variable 'SUFFIXES'" in invalid.value.message
                else:
                    assert await client.run(stages=["reload/change"], force=True)
            finally:
                await client.disconnect()
            if not refused:
                await recorder.wait_for(_helper_completes("reload/change"))
                await recorder.wait_for(_helper_is_idle)
            tg.cancel_scope.cancel()

    completions = [e for e in recorder.events if e["type"] == "stage_completed"]
    if refused:
        assert completions == [], "nothing runs: the early run was refused, not queued"
    else:
        assert [e["stage"] for e in completions] == ["reload/change"], (
            "only the run sent after the snapshot ran; the early one was not queued"
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("broken_source", "expected_error"),
    [
        pytest.param("def broken(:\n", "Failed to load", id="syntax-error"),
        pytest.param("value = 1\n", "No pipeline found during reload", id="no-pipeline-variable"),
    ],
)
async def test_broken_pipeline_edit_runs_nothing_until_fixed(
    watch_project: pathlib.Path,
    broken_source: str,
    expected_error: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken edit reports its error and runs nothing, even on data changes, until fixed."""
    caplog.set_level(logging.DEBUG, logger="pivot.engine.engine")
    pipeline_py = watch_project / "pipeline.py"
    pipeline_py.write_text(_helper_pipeline_source("-v1", third="removed"))
    pipeline = _helper_discover(watch_project)
    removed_output = _helper_output_path(watch_project, pipeline, "reload/removed")
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            await _helper_wait_until_watching(eng)

            pipeline_py.write_text(broken_source)
            invalid_index = await recorder.wait_for(_helper_is_invalid_pipeline)
            invalid = recorder.events[invalid_index]
            assert invalid["type"] == "pipeline_reloaded"
            assert invalid["error"] is not None
            assert expected_error in invalid["error"]
            assert sorted(invalid["stages"]) == [
                "reload/change",
                "reload/keep",
                "reload/removed",
            ], "an invalid edit keeps the last good pipeline's stages"
            assert eng.pipeline_error == invalid["error"]

            (watch_project / "data" / "raw" / "input.txt").write_text("y")
            with anyio.fail_after(30):
                while "ignoring run request" not in caplog.text:
                    await anyio.sleep(0.05)
            assert not removed_output.exists(), "an invalid pipeline runs nothing on a data change"

            pipeline_py.write_text(_helper_pipeline_source("-v1", third="added"))
            reload_index = await recorder.wait_for(_helper_is_reload, start=invalid_index)
            added_done = await recorder.wait_for(
                _helper_completes("reload/added"), start=reload_index
            )
            await recorder.wait_for(_helper_is_idle, start=added_done)

            assert eng.pipeline_error is None, "a good reload clears the invalid state"
            assert not any(
                event["type"] == "stage_completed" for event in recorder.events[:reload_index]
            ), "nothing ran while the pipeline was invalid"
            reload_event = recorder.events[reload_index]
            assert reload_event["type"] == "pipeline_reloaded"
            assert reload_event["stages_added"] == ["reload/added"], (
                "diffed against the last good pipeline"
            )
            assert reload_event["stages_removed"] == ["reload/removed"]
            assert recorder.completed("reload/added", start=reload_index) == StageStatus.RAN
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_watch_started_on_refused_pipeline_waits_for_fix(
    watch_project: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stage refused by fingerprinting at startup leaves the engine watching, not crashed."""
    pipeline_py = watch_project / "pipeline.py"
    pipeline_py.write_text(_helper_pipeline_source("-v1", third="other", refused=True))
    pipeline = _helper_discover(watch_project)
    change_output = _helper_output_path(watch_project, pipeline, "reload/change")
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        eng.add_source(
            sources.OneShotSource(stages=None, force=False, reason="test", max_workers=1)
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            await _helper_wait_until_watching(eng)
            invalid = recorder.events[await recorder.wait_for(_helper_is_invalid_pipeline)]
            assert invalid["type"] == "pipeline_reloaded"
            assert invalid["error"] is not None
            assert "captures non-scalar variable 'SUFFIXES'" in invalid["error"]
            assert "Pipeline invalid - waiting for fix" in caplog.text
            assert not any(event["type"] == "stage_completed" for event in recorder.events), (
                "no stage runs while the pipeline is invalid"
            )

            rpc_source = agent_rpc.AgentRpcSource(
                socket_path=watch_project / "unused.sock",
                handler=agent_rpc.AgentRpcHandler(engine=eng),
            )
            commit_send, _ = anyio.create_memory_object_stream[InputEvent](1)
            commit_response = await rpc_source._handle_request(
                {"jsonrpc": "2.0", "method": "commit", "id": 1}, commit_send
            )
            assert commit_response is not None
            assert "error" in commit_response, "RPC commit on a refused pipeline gets a reply"
            assert commit_response["error"]["code"] == -32003
            assert "captures non-scalar variable 'SUFFIXES'" in commit_response["error"]["message"]

            pipeline_py.write_text(_helper_pipeline_source("-v2", third="other"))
            reload_index = await recorder.wait_for(_helper_is_reload)
            done = await recorder.wait_for(_helper_completes("reload/change"), start=reload_index)
            await recorder.wait_for(_helper_is_idle, start=done)
            assert recorder.completed("reload/change", start=reload_index) == StageStatus.RAN
            assert change_output.read_text() == "x-v2", "the fixed pipeline runs"
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_refusal_added_mid_session_waits_for_fix(
    watch_project: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A refused edit mid-session is invalid like one at startup: run, commit and data refused.

    A commit while invalid would pair the edited input with the output of the old code and
    make the fixing reload skip the stage, so it must be refused too.
    """
    caplog.set_level(logging.DEBUG, logger="pivot.engine.engine")
    pipeline_py = watch_project / "pipeline.py"
    pipeline_py.write_text(_helper_pipeline_source("-v1", third="other"))
    pipeline = _helper_discover(watch_project)
    change_output = _helper_output_path(watch_project, pipeline, "reload/change")
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        eng.add_source(
            sources.OneShotSource(stages=None, force=False, reason="test", max_workers=1)
        )
        rpc_source = agent_rpc.AgentRpcSource(
            socket_path=watch_project / "unused.sock",
            handler=agent_rpc.AgentRpcHandler(engine=eng),
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            first_idle = await recorder.wait_for(_helper_is_idle)
            assert change_output.read_text() == "x-v1"

            pipeline_py.write_text(_helper_pipeline_source("-v2", third="other", refused=True))
            invalid_index = await recorder.wait_for(_helper_is_invalid_pipeline, start=first_idle)
            invalid = recorder.events[invalid_index]
            assert invalid["type"] == "pipeline_reloaded"
            assert invalid["error"] is not None
            assert "captures non-scalar variable 'SUFFIXES'" in invalid["error"]
            assert "Pipeline invalid - waiting for fix" in caplog.text
            assert eng.pipeline_error == invalid["error"], "a mid-session refusal is invalid too"

            run_send, run_recv = anyio.create_memory_object_stream[InputEvent](1)
            response = await rpc_source._handle_request(
                {"jsonrpc": "2.0", "method": "run", "params": {"force": True}, "id": 1}, run_send
            )
            assert response is not None
            assert "error" in response, "RPC run is refused, not accepted and dropped"
            assert response["error"]["code"] == -32003
            with pytest.raises(anyio.WouldBlock):
                run_recv.receive_nowait()

            (watch_project / "data" / "raw" / "input.txt").write_text("y")
            with anyio.fail_after(30):
                while "ignoring run request" not in caplog.text:
                    await anyio.sleep(0.05)
            assert change_output.read_text() == "x-v1", "the last good pipeline does not run"

            commit_response = await rpc_source._handle_request(
                {"jsonrpc": "2.0", "method": "commit", "id": 2}, run_send
            )
            assert commit_response is not None
            assert "error" in commit_response, "RPC commit is refused while invalid"
            assert commit_response["error"]["code"] == -32003
            assert "Pipeline invalid - waiting for fix" in commit_response["error"]["message"]

            pipeline_py.write_text(_helper_pipeline_source("-v3", third="other"))
            reload_index = await recorder.wait_for(_helper_is_reload, start=invalid_index)
            done = await recorder.wait_for(_helper_completes("reload/change"), start=reload_index)
            await recorder.wait_for(_helper_is_idle, start=done)

            assert eng.pipeline_error is None, "a good reload clears the invalid state"
            assert not any(
                event["type"] == "stage_completed"
                for event in recorder.events[invalid_index:reload_index]
            ), "nothing ran while the pipeline was invalid"
            reload_event = recorder.events[reload_index]
            assert reload_event["type"] == "pipeline_reloaded"
            assert reload_event["stages_modified"] == ["reload/change"]
            assert recorder.completed("reload/change", start=reload_index) == StageStatus.RAN, (
                "the fix reruns the stage on the data edited while invalid"
            )
            assert change_output.read_text() == "y-v3", "the fixed pipeline runs on the new input"
            tg.cancel_scope.cancel()


_PARAMS_PIPELINE = """\
from __future__ import annotations

from pivot import Pipeline, StageParams, stage


class SuffixParams(StageParams):
    suffix: str = "-default"


@stage
def tag(raw: str, params: SuffixParams) -> str:
    return raw + params.suffix


with Pipeline("reload") as pipeline:
    tag(pipeline.input("input.txt"), params=SuffixParams())
"""


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("bad_params", "expected_error"),
    [
        (
            "reload/tga:\n  suffix: -bad\n",
            "params.yaml key 'reload/tga' names no stage in pipeline 'reload'; "
            + "did you mean 'reload/tag'?",
        ),
        (
            "reload/tag:\n  suffix: 3\n",
            "params.yaml key 'reload/tag': field 'suffix' value 3 is invalid for SuffixParams: "
            + "Input should be a valid string",
        ),
        (
            "reload/tag:\n  sufix: -bad\n",
            "params.yaml key 'reload/tag': field 'sufix' is not a field of SuffixParams; "
            + "did you mean 'suffix'?",
        ),
    ],
    ids=["unknown-key", "bad-value", "unknown-field"],
)
async def test_watch_started_on_bad_params_yaml_waits_for_fix(
    watch_project: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    bad_params: str,
    expected_error: str,
) -> None:
    """A params.yaml error at startup is an invalid pipeline: nothing runs or commits until fixed."""
    (watch_project / "pipeline.py").write_text(_PARAMS_PIPELINE)
    params_yaml = watch_project / "params.yaml"
    params_yaml.write_text(bad_params)
    pipeline = _helper_discover(watch_project)
    tag_output = _helper_output_path(watch_project, pipeline, "reload/tag")
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        eng.add_source(
            sources.OneShotSource(stages=None, force=False, reason="test", max_workers=1)
        )
        rpc_source = agent_rpc.AgentRpcSource(
            socket_path=watch_project / "unused.sock",
            handler=agent_rpc.AgentRpcHandler(engine=eng),
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            await _helper_wait_until_watching(eng)
            invalid_index = await recorder.wait_for(_helper_is_invalid_pipeline)
            invalid = recorder.events[invalid_index]
            assert invalid["type"] == "pipeline_reloaded"
            assert invalid["error"] is not None
            assert expected_error in invalid["error"]
            assert eng.pipeline_error == invalid["error"]
            assert "Pipeline invalid - waiting for fix" in caplog.text

            run_send, run_recv = anyio.create_memory_object_stream[InputEvent](1)
            for request_id, method in enumerate(("run", "commit"), start=1):
                response = await rpc_source._handle_request(
                    {"jsonrpc": "2.0", "method": method, "id": request_id}, run_send
                )
                assert response is not None
                assert "error" in response, f"RPC {method} is refused while params.yaml is bad"
                assert response["error"]["code"] == -32003
                assert expected_error in response["error"]["message"]
            with pytest.raises(anyio.WouldBlock):
                run_recv.receive_nowait()
            assert not any(event["type"] == "stage_completed" for event in recorder.events)
            assert not tag_output.exists()

            params_yaml.write_text("reload/tag:\n  suffix: -fixed\n")
            reload_index = await recorder.wait_for(_helper_is_reload, start=invalid_index)
            done = await recorder.wait_for(_helper_completes("reload/tag"), start=reload_index)
            await recorder.wait_for(_helper_is_idle, start=done)
            assert eng.pipeline_error is None, "a good params.yaml clears the invalid state"
            assert recorder.completed("reload/tag", start=reload_index) == StageStatus.RAN
            assert tag_output.read_text() == "x-fixed"
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_run_that_finds_params_yaml_broken_marks_the_pipeline_invalid(
    watch_project: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An edit the watcher has not reported yet is caught by the run's load: watch stays up."""
    (watch_project / "pipeline.py").write_text(_PARAMS_PIPELINE)
    pipeline = _helper_discover(watch_project)
    tag_output = _helper_output_path(watch_project, pipeline, "reload/tag")
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(
            sources.FilesystemSource(watch_paths=[watch_project / "data" / "raw"], debounce_ms=50)
        )
        eng.add_source(
            sources.OneShotSource(stages=None, force=False, reason="test", max_workers=1)
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            first_idle = await recorder.wait_for(_helper_is_idle)
            assert tag_output.read_text() == "x-default"

            (watch_project / "params.yaml").write_text("reload/tga:\n  suffix: -bad\n")
            (watch_project / "data" / "raw" / "input.txt").write_text("y")
            invalid_index = await recorder.wait_for(_helper_is_invalid_pipeline, start=first_idle)
            invalid = recorder.events[invalid_index]
            assert invalid["type"] == "pipeline_reloaded"
            assert invalid["error"] is not None
            assert "params.yaml key 'reload/tga' names no stage" in invalid["error"]
            assert eng.pipeline_error == invalid["error"]
            assert "Pipeline invalid - waiting for fix" in caplog.text
            await recorder.wait_for(_helper_is_idle, start=invalid_index)
            assert tag_output.read_text() == "x-default", "no stage runs on a broken params.yaml"
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_reload_adding_a_stage_with_a_missing_input_waits_for_the_file(
    watch_project: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A reload whose re-run finds an input missing leaves watch up with the pipeline invalid.

    No stage runs while the input is absent. A write to another existing input is ignored; the
    missing input itself, or a directory that could contain it, re-checks and resumes the run.
    """
    pipeline_py = watch_project / "pipeline.py"
    pipeline_py.write_text(_helper_pipeline_source("-v1", third="other"))
    pipeline = _helper_discover(watch_project)
    change_output = _helper_output_path(watch_project, pipeline, "reload/change")
    recorder = _Recorder()
    caplog.set_level(logging.DEBUG, logger="pivot.engine.engine")

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        eng.add_source(
            sources.OneShotSource(stages=None, force=False, reason="test", max_workers=1)
        )
        rpc_source = agent_rpc.AgentRpcSource(
            socket_path=watch_project / "unused.sock",
            handler=agent_rpc.AgentRpcHandler(engine=eng),
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            first_idle = await recorder.wait_for(_helper_is_idle)
            assert change_output.read_text() == "x-v1"

            # Edits `change` and adds `late`, which reads an input that does not exist yet
            pipeline_py.write_text(
                _helper_pipeline_source("-v2", third="late").replace(
                    "    late(raw)\n", '    late(pipeline.input("missing.txt"))\n'
                )
            )
            invalid_index = await recorder.wait_for(_helper_is_invalid_pipeline, start=first_idle)
            invalid = recorder.events[invalid_index]
            assert invalid["type"] == "pipeline_reloaded"
            assert invalid["error"] is not None
            assert (
                "Stage 'reload/late' depends on input 'missing.txt' (path data/raw/missing.txt), "
                + "which does not exist"
            ) in invalid["error"]
            assert eng.pipeline_error == invalid["error"]
            assert (
                "Pipeline invalid: Stage 'reload/late' depends on input 'missing.txt'"
                in caplog.text
            )
            assert "Pipeline invalid - waiting for fix" in caplog.text
            await recorder.wait_for(_helper_is_idle, start=invalid_index)

            run_send, run_recv = anyio.create_memory_object_stream[InputEvent](1)
            response = await rpc_source._handle_request(
                {"jsonrpc": "2.0", "method": "run", "id": 1}, run_send
            )
            assert response is not None
            assert "error" in response, "RPC run is refused while the input is missing"
            assert response["error"]["code"] == -32003
            with pytest.raises(anyio.WouldBlock):
                run_recv.receive_nowait()

            event_start = len(recorder.events)
            log_start = len(caplog.records)
            (watch_project / "data" / "raw" / "input.txt").write_text("y")
            with anyio.fail_after(5):
                while not any(
                    "ignoring run request" in record.getMessage()
                    for record in caplog.records[log_start:]
                ) and not any(
                    event["type"] == "pipeline_reloaded" for event in recorder.events[event_start:]
                ):
                    await anyio.sleep(0.05)
            assert any(
                "ignoring run request" in record.getMessage()
                for record in caplog.records[log_start:]
            )
            assert not any(
                event["type"] == "pipeline_reloaded" for event in recorder.events[event_start:]
            ), "an existing input does not re-check a missing input"
            assert eng.pipeline_error == invalid["error"], "the input is still missing"

            (watch_project / "data" / "raw" / "missing.txt").write_text("m")
            valid_index = await recorder.wait_for(
                _helper_is_reload,
                start=invalid_index + 1,
            )
            late_done = await recorder.wait_for(_helper_completes("reload/late"), start=valid_index)
            await recorder.wait_for(_helper_is_idle, start=late_done)

            assert eng.pipeline_error is None, "the input's arrival makes the pipeline valid"
            assert not any(
                event["type"] == "stage_completed"
                for event in recorder.events[first_idle:valid_index]
            ), "nothing ran while the input was missing"
            completed = [
                event["stage"] for event in recorder.events if event["type"] == "stage_completed"
            ]
            assert completed.count("reload/late") == 1, "the new stage runs once"
            assert recorder.completed("reload/late", start=valid_index) == StageStatus.RAN
            assert recorder.completed("reload/change", start=valid_index) == StageStatus.RAN, (
                "the stopped re-run runs the stage the reload edited"
            )
            assert change_output.read_text() == "y-v2", "on the input edited while waiting"
            assert change_output.with_stem("late").read_text() == "m-late"
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_stage_completed_output_identity_reaches_tui_as_artifact_identity(
    watch_project: pathlib.Path,
) -> None:
    """``output_summary`` identities cross the RPC wire as ``{producer, key}`` objects."""
    (watch_project / "pipeline.py").write_text(_helper_pipeline_source("-v1", third="other"))
    pipeline = _helper_discover(watch_project)
    identity = pipeline.get_stage("reload/change")["outs"][0].identity
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(
            sources.OneShotSource(
                stages=["reload/change"], force=False, reason="test", max_workers=1
            )
        )
        await eng.run()

    completed = next(
        event
        for event in recorder.events
        if event["type"] == "stage_completed" and event["stage"] == "reload/change"
    )
    wire = json.loads(agent_rpc._encode_wire(completed))
    assert wire["output_summary"][0]["identity"] == {
        "producer": identity.producer,
        "key": identity.key,
    }

    msg = event_poller._make_completed_message(wire)
    assert msg is not None
    assert "output_summary" in msg
    summary = msg["output_summary"]
    assert summary is not None
    assert summary[0]["identity"] == identity


@pytest.mark.anyio
async def test_missing_input_that_arrives_with_its_directory_makes_the_pipeline_valid(
    watch_project: pathlib.Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A missing input whose directory does not exist yet, then arrives with the file inside it.

    Moving in a directory tree (mv, an unpacked archive, a download) shows the watcher the new
    directory only; the file inside it never gets an event of its own. The arrival must still
    count as the input's creation: the pipeline becomes valid and the stopped re-run runs.
    """
    pipeline_py = watch_project / "pipeline.py"
    pipeline_py.write_text(_helper_pipeline_source("-v1", third="other"))
    pipeline = _helper_discover(watch_project)
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        eng.add_source(
            sources.OneShotSource(stages=None, force=False, reason="test", max_workers=1)
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            first_idle = await recorder.wait_for(_helper_is_idle)

            # Adds `late`, which reads newdir/sub/deep.txt; newdir does not exist yet
            pipeline_py.write_text(
                _helper_pipeline_source("-v1", third="late").replace(
                    "    late(raw)\n",
                    '    late(pipeline.input("deep.txt", path="newdir/sub/deep.txt"))\n',
                )
            )
            invalid_index = await recorder.wait_for(_helper_is_invalid_pipeline, start=first_idle)
            invalid = recorder.events[invalid_index]
            assert invalid["type"] == "pipeline_reloaded"
            assert invalid["error"] is not None
            assert "(path newdir/sub/deep.txt), which does not exist" in invalid["error"]
            idle_index = await recorder.wait_for(_helper_is_idle, start=invalid_index)

            # The input arrives with its directory, as `mv`, `cp -r` or `tar -x` deliver it
            staging = tmp_path_factory.mktemp("staging")
            (staging / "newdir" / "sub").mkdir(parents=True)
            (staging / "newdir" / "sub" / "deep.txt").write_text("d")
            os.rename(staging / "newdir", watch_project / "newdir")

            valid_index = await recorder.wait_for(
                _helper_is_reload,
                start=idle_index,
                timeout=30.0,
            )
            await recorder.wait_for(
                _helper_completes("reload/late"), start=valid_index, timeout=30.0
            )
            assert recorder.completed("reload/late", start=valid_index) == StageStatus.RAN
            assert eng.pipeline_error is None, "the input's arrival makes the pipeline valid"
            tg.cancel_scope.cancel()


class _FileSink:
    """Sink that appends each event's type to a file, as ``--jsonl > <file>`` writes it."""

    _path: pathlib.Path

    def __init__(self, path: pathlib.Path) -> None:
        self._path = path

    async def handle(self, event: OutputEvent) -> None:
        with self._path.open("a") as handle:
            handle.write(f"{event['type']}\n")

    async def close(self) -> None:
        pass


@pytest.mark.anyio
async def test_output_written_inside_the_project_does_not_recheck_a_missing_input(
    watch_project: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """While an input is missing, the session's own output inside the project re-checks nothing.

    A session whose output lands in the watched root (``--jsonl > events.jsonl``, a log file)
    writes on every re-check. Only a change at a missing input's path, or at a directory that
    would contain it, checks the inputs again.
    """
    caplog.set_level(logging.INFO, logger="pivot.engine.engine")
    pipeline_py = watch_project / "pipeline.py"
    pipeline_py.write_text(_helper_pipeline_source("-v1", third="other"))
    pipeline = _helper_discover(watch_project)
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_sink(_FileSink(watch_project / "events.jsonl"))
        eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
        eng.add_source(
            sources.OneShotSource(stages=None, force=False, reason="test", max_workers=1)
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            first_idle = await recorder.wait_for(_helper_is_idle)

            pipeline_py.write_text(
                _helper_pipeline_source("-v1", third="late").replace(
                    "    late(raw)\n", '    late(pipeline.input("missing.txt"))\n'
                )
            )
            invalid_index = await recorder.wait_for(_helper_is_invalid_pipeline, start=first_idle)
            idle_index = await recorder.wait_for(_helper_is_idle, start=invalid_index)
            log_offset = len(caplog.text)

            await anyio.sleep(10)
            rechecks = [
                event
                for event in recorder.events[idle_index:]
                if event["type"] == "pipeline_reloaded"
            ]
            assert not rechecks, f"{len(rechecks)} re-checks fed by the session's own output"
            assert "checking again" not in caplog.text[log_offset:]

            (watch_project / "data" / "raw" / "missing.txt").write_text("m")
            valid_index = await recorder.wait_for(
                _helper_is_reload,
                start=idle_index,
            )
            await recorder.wait_for(_helper_completes("reload/late"), start=valid_index)
            assert recorder.completed("reload/late", start=valid_index) == StageStatus.RAN
            tg.cancel_scope.cancel()


_OWNERSHIP_BEFORE = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage
def keep(raw: str) -> str:
    return raw + "-keep"


with Pipeline("reload") as pipeline:
    keep(pipeline.input("input.txt"))
"""

# Adds `tree`, whose directory output data/reload/tree/ holds the tracked f.txt.pvt
_OWNERSHIP_AFTER = """\
from __future__ import annotations

from typing import Annotated

from pivot.compose import Pipeline, directory, stage


@stage
def keep(raw: str) -> str:
    return raw + "-keep"


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {"g.txt": "g"}


with Pipeline("reload") as pipeline:
    keep(pipeline.input("input.txt"))
    tree()
"""


def _helper_leaf_errors(
    group: BaseExceptionGroup[exceptions.TrackedFileMissingError],
) -> list[exceptions.TrackedFileMissingError]:
    """The errors in ``group`` and in the groups nested in it (each task group adds a level)."""
    leaves = list[exceptions.TrackedFileMissingError]()
    for error in group.exceptions:
        if isinstance(error, BaseExceptionGroup):
            leaves.extend(_helper_leaf_errors(error))
        else:
            leaves.append(error)
    return leaves


@pytest.mark.anyio
async def test_reload_declaring_the_output_around_a_missing_tracked_file_keeps_its_sidecar(
    watch_project: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """Audit D1, watch reload adding ownership: the re-run stops and the .pvt survives.

    A file tracked while no stage declared its directory is missing. An edit to pipeline.py
    that declares that directory as a stage's output re-runs the pipeline; the re-run must stop
    on the missing tracked file rather than run the stage, whose output replacement would
    delete the sidecar.
    """
    pipeline_py = watch_project / "pipeline.py"
    pipeline_py.write_text(_OWNERSHIP_BEFORE)
    tracked = watch_project / "data" / "reload" / "tree" / "f.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("tracked bytes")
    track_result = runner.invoke(cli.cli, ["track", "data/reload/tree/f.txt"])
    assert track_result.exit_code == 0, track_result.output
    pvt_path = track.get_pvt_path(tracked)
    sidecar = pvt_path.read_bytes()
    tracked.unlink()
    pipeline = _helper_discover(watch_project)
    recorder = _Recorder()
    missing_errors = list[exceptions.TrackedFileMissingError]()

    try:
        async with engine_mod.Engine(pipeline=pipeline) as eng:
            eng.add_sink(recorder)
            eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
            async with anyio.create_task_group() as tg:
                tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
                await _helper_wait_until_watching(eng)

                pipeline_py.write_text(_OWNERSHIP_AFTER)
                reload_index = await recorder.wait_for(_helper_is_reload)
                # The missing-file error ends eng.run and cancels this wait; reaching the
                # stage's completion means the re-run ran it instead
                tree_done = await recorder.wait_for(
                    _helper_completes("reload/tree"), start=reload_index
                )
                await recorder.wait_for(_helper_is_idle, start=tree_done)
                tg.cancel_scope.cancel()
    except* exceptions.TrackedFileMissingError as group:
        missing_errors.extend(_helper_leaf_errors(group))

    assert pvt_path.is_file(), "the reload's re-run must not delete the tracked file's .pvt"
    assert pvt_path.read_bytes() == sidecar
    assert not any(
        event["type"] == "stage_completed" and event["stage"] == "reload/tree"
        for event in recorder.events
    ), "the stage owning the missing tracked file must not run"
    assert len(missing_errors) == 1, "the re-run must stop on the missing tracked file"
    assert "f.txt" in str(missing_errors[0])


_RPC_ROOT_ALONE = """\
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("a") as pipeline:
    alive()
"""

_RPC_ROOT_INCLUDING_B = """\
from other.pipeline import pipeline as b
from pivot import Pipeline, stage


@stage
def alive() -> str:
    return "alive"


with Pipeline("a") as pipeline:
    alive()
    pipeline.include(b)
"""

# Pipeline b's stage writes a directory output at data/b/tree/
_RPC_OTHER = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {"g.txt": "g"}


with Pipeline("b") as pipeline:
    tree()
"""


@pytest.mark.anyio
async def test_rpc_forced_run_stops_on_a_missing_file_tracked_inside_another_pipelines_output(
    watch_project: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """Audit D1, watch RPC row: a forced RPC `run` of a/alive stops on the missing tracked file.

    The file was tracked inside b/tree's directory output while only pipeline a was loaded. With
    a/alive and b/tree loaded in a watching engine, an RPC `run` with `force` must not run the
    stage while that tracked file is missing.
    """
    (watch_project / "pipeline.py").write_text(_RPC_ROOT_ALONE)
    (watch_project / "other").mkdir()
    (watch_project / "other" / "pipeline.py").write_text(_RPC_OTHER)
    tracked = watch_project / "data" / "b" / "tree" / "f.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("tracked bytes")
    track_result = runner.invoke(cli.cli, ["track", "data/b/tree/f.txt"])
    assert track_result.exit_code == 0, track_result.output
    pvt_path = track.get_pvt_path(tracked)
    tracked.unlink()
    (watch_project / "pipeline.py").write_text(_RPC_ROOT_INCLUDING_B)
    pipeline = _helper_discover(watch_project)
    assert sorted(pipeline.list_stages()) == ["a/alive", "b/tree"]
    recorder = _Recorder()
    socket_path = watch_project / "rpc.sock"
    ready = threading.Event()
    accepted = list[bool]()
    missing_errors = list[exceptions.TrackedFileMissingError]()

    try:
        async with engine_mod.Engine(pipeline=pipeline) as eng:
            eng.add_sink(recorder)
            handler = agent_rpc.AgentRpcHandler(engine=eng)
            eng.add_source(
                agent_rpc.AgentRpcSource(socket_path=socket_path, handler=handler, ready=ready)
            )
            eng.add_source(sources.FilesystemSource(watch_paths=[watch_project], debounce_ms=50))
            async with anyio.create_task_group() as tg:
                tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
                await _helper_wait_until_watching(eng)
                client = rpc_client_impl.RpcPivotClient()
                with anyio.fail_after(5):
                    while not ready.is_set():
                        await anyio.sleep(0.01)
                    await client.connect(socket_path)
                try:
                    accepted.append(await client.run(["a/alive"], force=True))
                    # The missing-file error ends eng.run and cancels this wait; reaching the
                    # stage's completion means the run ran it instead
                    alive_done = await recorder.wait_for(_helper_completes("a/alive"))
                    await recorder.wait_for(_helper_is_idle, start=alive_done)
                finally:
                    with anyio.CancelScope(shield=True):
                        await client.disconnect()
                tg.cancel_scope.cancel()
    except* exceptions.TrackedFileMissingError as group:
        missing_errors.extend(_helper_leaf_errors(group))

    assert accepted == [True], "the engine accepts the RPC run request"
    assert not any(
        event["type"] == "stage_completed" and event["stage"] == "a/alive"
        for event in recorder.events
    ), "the forced run must not run a/alive while a tracked file is missing"
    assert len(missing_errors) == 1, "the run must stop on the missing tracked file"
    assert "data/b/tree/f.txt" in str(missing_errors[0])
    assert pvt_path.is_file(), "the stopped run keeps the sidecar"
