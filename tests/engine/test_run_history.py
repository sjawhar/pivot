"""Tests for run history after Engine execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot import config
from pivot.engine import engine
from pivot.storage import state as state_mod
from tests import helpers

if TYPE_CHECKING:
    import pathlib


def _helper_stage_func(params: None) -> dict[str, str]:
    """Simple stage with no deps or outputs for testing."""
    return {"result": "success"}


@pytest.fixture
def registered_stage() -> str:
    """Register a simple stage for testing."""
    helpers.register_test_stage(
        func=_helper_stage_func,
        name="history_test",
    )
    return "history_test"


def test_engine_writes_run_history(
    registered_stage: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine writes run history after execution."""
    # Set up paths
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / "state"
    monkeypatch.setattr(config, "get_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(config, "get_state_dir", lambda: state_dir)
    monkeypatch.setattr(config, "get_state_db_path", lambda: state_dir / "state.db")

    eng = engine.Engine()

    # Run the stage
    results = eng.run_once(
        stages=[registered_stage],
        cache_dir=cache_dir,
    )

    assert registered_stage in results

    # Verify run history was written
    with state_mod.StateDB(state_dir / "state.db") as state_db:
        runs = state_db.list_runs(limit=1)

    assert len(runs) >= 1

    latest = runs[0]
    assert "run_id" in latest
    assert "started_at" in latest
    assert "ended_at" in latest


def test_engine_run_history_contains_stage_records(
    registered_stage: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine run history contains records for executed stages."""
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / "state"
    monkeypatch.setattr(config, "get_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(config, "get_state_dir", lambda: state_dir)
    monkeypatch.setattr(config, "get_state_db_path", lambda: state_dir / "state.db")

    eng = engine.Engine()

    # Run the stage
    eng.run_once(stages=[registered_stage], cache_dir=cache_dir)

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


def test_engine_writes_run_cache_entry(
    registered_stage: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine writes run cache entries for successful stages."""
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / "state"
    monkeypatch.setattr(config, "get_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(config, "get_state_dir", lambda: state_dir)
    monkeypatch.setattr(config, "get_state_db_path", lambda: state_dir / "state.db")

    eng = engine.Engine()

    # Run twice - second should be cached
    eng.run_once(stages=[registered_stage], cache_dir=cache_dir)
    results = eng.run_once(stages=[registered_stage], cache_dir=cache_dir)

    # Should be skipped due to cache
    assert results[registered_stage]["reason"] != "", "Stage should have a skip reason"
