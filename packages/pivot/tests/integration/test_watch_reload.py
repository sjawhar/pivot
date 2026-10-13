"""Watch-mode code reload through a running Engine, driven by real pipeline.py edits.

Each test writes a project, discovers its pipeline the way the CLI does, runs an Engine with a
FilesystemSource on the project root (the wiring ``pivot repro --watch`` uses), edits pipeline.py
on disk and observes the engine's output events and the files its stages write.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import anyio
import pytest

from pivot import discovery, project
from pivot.engine import engine as engine_mod
from pivot.engine import sources
from pivot.engine.types import EngineState, OutputEvent
from pivot.storage import store as store_mod
from pivot.types import StageStatus

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    from pivot.registry import PipelineLike


def _helper_pipeline_source(change_suffix: str, *, third: str) -> str:
    """pipeline.py with stages keep, change (input + change_suffix) and ``third``."""
    return f"""\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage
def keep(raw: str) -> str:
    return raw + "-keep"


@stage
def change(raw: str) -> str:
    return raw + {change_suffix!r}


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


def _helper_is_idle(event: OutputEvent) -> bool:
    return event["type"] == "engine_state_changed" and event["state"] == EngineState.IDLE


def _helper_is_reload(event: OutputEvent) -> bool:
    return event["type"] == "pipeline_reloaded"


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
@pytest.mark.xfail(
    strict=True,
    reason="red test for watch-fixes (Task 26): reload fingerprints the old pipeline lazily, "
    + "after the edit, so an edited stage never fingerprinted before is not reported modified",
)
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
            await anyio.wait_all_tasks_blocked()

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
@pytest.mark.parametrize(
    "broken_source",
    [
        pytest.param("def broken(:\n", id="syntax-error"),
        pytest.param("value = 1\n", id="no-pipeline-variable"),
    ],
)
async def test_broken_pipeline_edit_keeps_last_good_pipeline_until_fixed(
    watch_project: pathlib.Path,
    broken_source: str,
) -> None:
    """A broken edit emits no reload; data changes still run the last good pipeline's stages."""
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
            await anyio.wait_all_tasks_blocked()

            # The engine handles the code event before the data event it observes after it.
            pipeline_py.write_text(broken_source)
            (watch_project / "data" / "raw" / "input.txt").write_text("y")
            removed_done = await recorder.wait_for(_helper_completes("reload/removed"))
            broken_idle = await recorder.wait_for(_helper_is_idle, start=removed_done)
            assert removed_output.read_text() == "y-removed", (
                "the last good pipeline's stage ran on the changed input"
            )

            pipeline_py.write_text(_helper_pipeline_source("-v1", third="added"))
            reload_index = await recorder.wait_for(_helper_is_reload, start=broken_idle)
            added_done = await recorder.wait_for(
                _helper_completes("reload/added"), start=reload_index
            )
            await recorder.wait_for(_helper_is_idle, start=added_done)

            assert not any(_helper_is_reload(event) for event in recorder.events[:reload_index]), (
                "a broken pipeline.py must not be reported as a reload"
            )
            assert recorder.completed("reload/removed", start=0) == StageStatus.RAN
            reload_event = recorder.events[reload_index]
            assert reload_event["type"] == "pipeline_reloaded"
            assert reload_event["stages_added"] == ["reload/added"], (
                "diffed against the last good pipeline"
            )
            assert reload_event["stages_removed"] == ["reload/removed"]
            assert recorder.completed("reload/added", start=reload_index) == StageStatus.RAN
            tg.cancel_scope.cancel()
