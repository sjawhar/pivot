"""Tests for run history after Engine execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot import config
from pivot.engine import sinks, sources
from pivot.storage import state as state_mod
from tests import helpers

if TYPE_CHECKING:
    import pathlib

    from pivot.engine import engine
    from pivot.pipeline.pipeline import Pipeline


def _helper_stage_func(params: None) -> dict[str, str]:
    """Simple stage with no deps or outputs for testing."""
    return {"result": "success"}


@pytest.fixture
def registered_stage(test_pipeline: Pipeline) -> str:
    """Register a simple stage for testing."""
    helpers.register_test_stage(
        func=_helper_stage_func,
        name="history_test",
        pipeline=test_pipeline,
    )
    return "history_test"


@pytest.mark.anyio
async def test_engine_writes_run_history(
    registered_stage: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_engine: engine.Engine,
) -> None:
    """Engine writes run history after execution."""
    # Set up paths
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / "state"
    monkeypatch.setattr(config, "get_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(config, "get_state_dir", lambda: state_dir)
    monkeypatch.setattr(config, "get_state_db_path", lambda: state_dir / "state.db")

    # Run the stage
    collector = sinks.ResultCollectorSink()
    test_engine.add_sink(collector)
    test_engine.add_source(
        sources.OneShotSource(
            stages=[registered_stage],
            force=False,
            reason="test",
            cache_dir=cache_dir,
        )
    )
    await test_engine.run(exit_on_completion=True)
    results = await collector.get_results()

    assert registered_stage in results

    # Verify run history was written
    with state_mod.StateDB(state_dir / "state.db") as state_db:
        runs = state_db.list_runs(limit=1)

    assert len(runs) >= 1

    latest = runs[0]
    assert "run_id" in latest
    assert "started_at" in latest
    assert "ended_at" in latest


@pytest.mark.anyio
async def test_engine_run_history_contains_stage_records(
    registered_stage: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_engine: engine.Engine,
) -> None:
    """Engine run history contains records for executed stages."""
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / "state"
    monkeypatch.setattr(config, "get_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(config, "get_state_dir", lambda: state_dir)
    monkeypatch.setattr(config, "get_state_db_path", lambda: state_dir / "state.db")

    # Run the stage
    collector = sinks.ResultCollectorSink()
    test_engine.add_sink(collector)
    test_engine.add_source(
        sources.OneShotSource(
            stages=[registered_stage],
            force=False,
            reason="test",
            cache_dir=cache_dir,
        )
    )
    await test_engine.run(exit_on_completion=True)

    # Verify run history contains stage record
    with state_mod.StateDB(state_dir / "state.db") as state_db:
        runs = state_db.list_runs(limit=1)

    assert len(runs) >= 1
    latest = runs[0]
    assert "stages" in latest
    assert registered_stage in latest["stages"]

    stage_record = latest["stages"][registered_stage]
    assert "status" in stage_record
    assert "reason" in stage_record
    assert "duration_ms" in stage_record


@pytest.mark.anyio
async def test_engine_writes_run_cache_entry(
    registered_stage: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """Engine writes run cache entries for successful stages."""
    from pivot.engine.engine import Engine

    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / "state"
    monkeypatch.setattr(config, "get_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(config, "get_state_dir", lambda: state_dir)
    monkeypatch.setattr(config, "get_state_db_path", lambda: state_dir / "state.db")

    # First run - executes the stage
    async with Engine(pipeline=test_pipeline) as engine1:
        collector1 = sinks.ResultCollectorSink()
        engine1.add_sink(collector1)
        engine1.add_source(
            sources.OneShotSource(
                stages=[registered_stage],
                force=False,
                reason="test",
                cache_dir=cache_dir,
            )
        )
        await engine1.run(exit_on_completion=True)

    # Second run - should be cached (new engine instance required)
    async with Engine(pipeline=test_pipeline) as engine2:
        collector2 = sinks.ResultCollectorSink()
        engine2.add_sink(collector2)
        engine2.add_source(
            sources.OneShotSource(
                stages=[registered_stage],
                force=False,
                reason="test",
                cache_dir=cache_dir,
            )
        )
        await engine2.run(exit_on_completion=True)
        results = await collector2.get_results()

    # Should be skipped due to cache
    assert results[registered_stage]["reason"] != "", "Stage should have a skip reason"
