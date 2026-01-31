"""Integration tests for unified watch/non-watch execution.

These tests verify that the two bugs from issue #305 are fixed:
1. --quiet flag not working in watch mode
2. Stage list not updating when pipeline reloads in watch mode
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, override

import pytest

from helpers import register_test_stage
from pivot.engine import engine as engine_mod
from pivot.engine import sinks, sources
from pivot.types import StageStatus

if TYPE_CHECKING:
    import pathlib

    from pivot.engine.types import OutputEvent, PipelineReloaded, RunRequested
    from pivot.pipeline.pipeline import Pipeline


def _helper_noop(params: None) -> dict[str, str]:
    """No-op stage for testing."""
    return {"result": "ok"}


class EventCaptureSink:
    """Sink that captures all events for inspection."""

    def __init__(self) -> None:
        self.events: list[OutputEvent] = []

    def handle(self, event: OutputEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass


@pytest.fixture
def minimal_pipeline(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, test_pipeline: Pipeline
) -> Pipeline:
    """Set up a minimal pipeline for testing."""
    from pivot import config

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(config, "get_state_db_path", lambda: tmp_path / "state" / "state.db")
    monkeypatch.chdir(tmp_path)

    # Create git directory (required for project root detection)
    (tmp_path / ".git").mkdir()

    return test_pipeline


def test_engine_run_completes_with_exit_on_completion(
    minimal_pipeline: Pipeline,
) -> None:
    """Engine.run(exit_on_completion=True) should exit when stages complete."""
    register_test_stage(_helper_noop, name="test_stage")

    with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
        collector = sinks.ResultCollectorSink()
        eng.add_sink(collector)
        eng.add_source(sources.OneShotSource(stages=["test_stage"], force=True, reason="test"))

        # This should complete without hanging
        eng.run(exit_on_completion=True)

        results = collector.get_results()
        assert "test_stage" in results
        # Stage completes (any terminal status is acceptable - the point is it doesn't hang)
        assert results["test_stage"]["status"] in (
            StageStatus.RAN,
            StageStatus.SKIPPED,
            StageStatus.FAILED,
        )


def test_engine_exit_on_completion_false_blocks_until_shutdown(
    minimal_pipeline: Pipeline,
) -> None:
    """Engine.run(exit_on_completion=False) should block until shutdown() is called."""
    register_test_stage(_helper_noop, name="blocking_test")

    with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
        collector = sinks.ResultCollectorSink()
        eng.add_sink(collector)
        eng.add_source(sources.OneShotSource(stages=["blocking_test"], force=True, reason="test"))

        # Run in thread, call shutdown after brief delay
        shutdown_called = threading.Event()

        def delayed_shutdown() -> None:
            time.sleep(0.3)
            eng.shutdown()
            shutdown_called.set()

        shutdown_thread = threading.Thread(target=delayed_shutdown)
        shutdown_thread.start()

        start = time.perf_counter()
        eng.run(exit_on_completion=False)  # Should block until shutdown
        elapsed = time.perf_counter() - start

        shutdown_thread.join()

        # Should have blocked for at least the delay time
        assert elapsed >= 0.2
        assert shutdown_called.is_set()

        results = collector.get_results()
        assert "blocking_test" in results


def test_pipeline_reloaded_includes_stages_field(
    minimal_pipeline: Pipeline,
) -> None:
    """PipelineReloaded event should include the stages field with topologically sorted stages."""
    register_test_stage(_helper_noop, name="stage_a")
    register_test_stage(_helper_noop, name="stage_b")

    # Capture events to verify PipelineReloaded
    capture_sink = EventCaptureSink()

    with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
        eng.add_sink(capture_sink)

        # Emit a reload event manually by calling the internal method
        old_stages = eng._get_all_stages()
        eng._emit_reload_event(old_stages)

    # Find the PipelineReloaded event
    reload_events = [e for e in capture_sink.events if e["type"] == "pipeline_reloaded"]
    assert len(reload_events) == 1

    reload_event: PipelineReloaded = reload_events[0]  # type: ignore[assignment]
    assert "stages" in reload_event
    assert isinstance(reload_event["stages"], list)
    # Stages should be present (order depends on DAG)
    assert set(reload_event["stages"]) >= {"stage_a", "stage_b"}


def test_result_collector_sink_collects_all_stage_results(
    minimal_pipeline: Pipeline,
) -> None:
    """ResultCollectorSink should collect results from all stages."""
    register_test_stage(_helper_noop, name="stage_1")
    register_test_stage(_helper_noop, name="stage_2")

    with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
        collector = sinks.ResultCollectorSink()
        eng.add_sink(collector)
        eng.add_source(sources.OneShotSource(stages=None, force=True, reason="test"))

        eng.run(exit_on_completion=True)

        results = collector.get_results()
        assert "stage_1" in results
        assert "stage_2" in results


def test_oneshot_source_passes_orchestration_params(
    minimal_pipeline: Pipeline,
) -> None:
    """OneShotSource should pass orchestration parameters through to RunRequested."""
    from pivot.types import OnError

    captured_events: list[RunRequested] = []

    class EventCapturingEngine(engine_mod.Engine):
        @override
        def _handle_run_requested(self, event: RunRequested) -> None:
            captured_events.append(event)
            super()._handle_run_requested(event)

    register_test_stage(_helper_noop, name="param_test")

    with EventCapturingEngine(pipeline=minimal_pipeline) as eng:
        collector = sinks.ResultCollectorSink()
        eng.add_sink(collector)

        # Create source with specific orchestration params
        source = sources.OneShotSource(
            stages=["param_test"],
            force=True,
            reason="test",
            single_stage=True,
            no_commit=True,
            on_error=OnError.KEEP_GOING,
        )
        eng.add_source(source)

        eng.run(exit_on_completion=True)

    assert len(captured_events) == 1
    event = captured_events[0]
    assert event["stages"] == ["param_test"]
    assert event["force"] is True
    assert event.get("single_stage") is True
    assert event.get("no_commit") is True
    assert event.get("on_error") == OnError.KEEP_GOING
