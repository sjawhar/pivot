from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from pivot import metrics

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest import MonkeyPatch


@pytest.fixture(autouse=True)
def reset_metrics() -> Generator[None]:
    """Reset metrics state before each test."""
    metrics.clear()
    original_enabled = metrics._enabled
    yield
    metrics.clear()
    metrics._enabled = original_enabled


# =============================================================================
# enable() tests
# =============================================================================


def test_enable_sets_flag():
    metrics._enabled = False
    metrics.enable()
    assert metrics._enabled is True


def test_enable_idempotent():
    metrics._enabled = True
    metrics.enable()
    assert metrics._enabled is True


# =============================================================================
# clear() tests
# =============================================================================


def test_clear_removes_all_entries():
    metrics._enabled = True
    _t = metrics.start()
    metrics.end("test", _t)
    assert len(metrics._durations) > 0
    metrics.clear()
    assert len(metrics._durations) == 0


def test_clear_on_empty():
    metrics.clear()
    assert len(metrics._durations) == 0


# =============================================================================
# get_entries() tests
# =============================================================================


def test_get_entries_empty():
    assert metrics.get_entries() == []


def test_get_entries_single_metric():
    metrics.add_entries([("test", 1.0), ("test", 2.0)])
    entries = metrics.get_entries()
    assert len(entries) == 2
    assert ("test", 1.0) in entries
    assert ("test", 2.0) in entries


def test_get_entries_multiple_metrics():
    metrics.add_entries([("a", 1.0), ("b", 2.0), ("a", 3.0)])
    entries = metrics.get_entries()
    assert len(entries) == 3
    names = [e[0] for e in entries]
    assert names.count("a") == 2
    assert names.count("b") == 1


# =============================================================================
# add_entries() tests
# =============================================================================


def test_add_entries_creates_metric():
    metrics.add_entries([("new_metric", 5.0)])
    assert "new_metric" in metrics._durations
    assert metrics._durations["new_metric"] == [5.0]


def test_add_entries_appends_to_existing():
    metrics.add_entries([("metric", 1.0)])
    metrics.add_entries([("metric", 2.0)])
    assert metrics._durations["metric"] == [1.0, 2.0]


def test_add_entries_empty_list():
    metrics.add_entries([])
    assert len(metrics._durations) == 0


# =============================================================================
# summary() tests
# =============================================================================


def test_summary_empty():
    assert metrics.summary() == {}


def test_summary_single_entry():
    metrics.add_entries([("test", 10.0)])
    result = metrics.summary()
    assert "test" in result
    assert result["test"]["count"] == 1.0
    assert result["test"]["total_ms"] == 10.0
    assert result["test"]["avg_ms"] == 10.0
    assert result["test"]["min_ms"] == 10.0
    assert result["test"]["max_ms"] == 10.0


def test_summary_multiple_entries():
    metrics.add_entries([("test", 10.0), ("test", 20.0), ("test", 30.0)])
    result = metrics.summary()
    assert result["test"]["count"] == 3.0
    assert result["test"]["total_ms"] == 60.0
    assert result["test"]["avg_ms"] == 20.0
    assert result["test"]["min_ms"] == 10.0
    assert result["test"]["max_ms"] == 30.0


def test_summary_multiple_metrics_sorted():
    metrics.add_entries([("z_metric", 1.0), ("a_metric", 2.0)])
    result = metrics.summary()
    keys = list(result.keys())
    assert keys == ["a_metric", "z_metric"]


# =============================================================================
# start()/end() tests
# =============================================================================


def test_start_end_disabled_does_not_record():
    metrics._enabled = False
    _t = metrics.start()
    metrics.end("test", _t)
    assert len(metrics._durations) == 0


def test_start_end_enabled_records_duration():
    metrics._enabled = True
    _t = metrics.start()
    time.sleep(0.01)  # 10ms
    metrics.end("test", _t)
    assert "test" in metrics._durations
    assert len(metrics._durations["test"]) == 1
    # Duration should be at least 10ms
    assert metrics._durations["test"][0] >= 10.0


def test_start_end_records_on_exception():
    metrics._enabled = True
    _t = metrics.start()
    try:
        raise ValueError("test error")
    except ValueError:
        pass
    finally:
        metrics.end("test", _t)
    # Metrics should still be recorded even when exception occurs
    assert "test" in metrics._durations
    assert len(metrics._durations["test"]) == 1


def test_start_end_nested():
    metrics._enabled = True
    _t_outer = metrics.start()
    _t_inner = metrics.start()
    metrics.end("inner", _t_inner)
    metrics.end("outer", _t_outer)
    assert "outer" in metrics._durations
    assert "inner" in metrics._durations


def test_start_end_same_name_accumulates():
    metrics._enabled = True
    _t = metrics.start()
    metrics.end("test", _t)
    _t = metrics.start()
    metrics.end("test", _t)
    assert len(metrics._durations["test"]) == 2


# =============================================================================
# MAX_ENTRIES trimming tests
# =============================================================================


def test_trimming_triggers_at_max_entries(monkeypatch: MonkeyPatch):
    # Use a smaller limit for testing
    monkeypatch.setattr(metrics, "MAX_ENTRIES", 10)

    # Add exactly MAX_ENTRIES
    for i in range(10):
        metrics.add_entries([("test", float(i))])

    total = sum(len(ds) for ds in metrics._durations.values())
    assert total == 10

    # Add one more to trigger trimming
    metrics.add_entries([("test", 10.0)])

    total = sum(len(ds) for ds in metrics._durations.values())
    # After trimming, should have ~half the entries (5 or 6 depending on implementation)
    assert total < 10


def test_trimming_preserves_recent_entries(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(metrics, "MAX_ENTRIES", 10)

    # Add 11 entries to trigger trim
    for i in range(11):
        metrics.add_entries([("test", float(i))])

    # After trimming, should have newer entries (higher values)
    # Trimming removes oldest half, so values 0-4 should be removed
    remaining = metrics._durations["test"]
    assert all(v >= 5.0 for v in remaining)


def test_trimming_multiple_metrics(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(metrics, "MAX_ENTRIES", 10)

    # Add entries across multiple metrics
    for i in range(6):
        metrics.add_entries([("a", float(i))])
    for i in range(6):
        metrics.add_entries([("b", float(i))])

    # Should have triggered trimming, both metrics reduced
    total = sum(len(ds) for ds in metrics._durations.values())
    assert total < 12


# =============================================================================
# Cross-process integration tests
# =============================================================================


def test_roundtrip_serialization():
    """Test that get_entries/add_entries can transfer metrics across processes."""
    metrics._enabled = True

    # Simulate worker: collect metrics
    _t = metrics.start()
    metrics.end("worker.task", _t)
    entries = metrics.get_entries()

    # Clear to simulate new process
    metrics.clear()
    assert metrics.get_entries() == []

    # Simulate main process: aggregate
    metrics.add_entries(entries)

    # Verify metrics transferred
    assert "worker.task" in metrics._durations
    result = metrics.summary()
    assert result["worker.task"]["count"] == 1.0
