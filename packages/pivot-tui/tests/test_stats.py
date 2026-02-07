from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from pivot_tui import stats

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# =============================================================================
# SlidingWindowCounter Tests
# =============================================================================


def test_sliding_window_counter_initial_throughput_is_zero() -> None:
    counter = stats.SlidingWindowCounter()
    assert counter.get_throughput() == 0.0


def test_sliding_window_counter_records_events() -> None:
    counter = stats.SlidingWindowCounter()
    counter.record()
    counter.record()
    counter.record()
    # With 3 events in the 5-second window, throughput should be 3/5 = 0.6
    throughput = counter.get_throughput()
    assert throughput == pytest.approx(0.6, abs=0.1)


def test_sliding_window_counter_throughput_reflects_window() -> None:
    counter = stats.SlidingWindowCounter()
    # Record events
    for _ in range(10):
        counter.record()
    throughput = counter.get_throughput()
    # 10 events in 5 seconds = 2/s
    assert throughput == pytest.approx(2.0, abs=0.5)


def test_sliding_window_counter_handles_bucket_transitions(mocker: MockerFixture) -> None:
    # Mock time.monotonic to control bucket transitions
    mock_time = mocker.patch("pivot_tui.stats.time.monotonic", autospec=True)

    counter = stats.SlidingWindowCounter()

    # Record events at time 0.05 (bucket 0.0)
    mock_time.return_value = 0.05
    counter.record()
    counter.record()

    # Move to time 0.15 (bucket 0.1 - new bucket)
    mock_time.return_value = 0.15
    counter.record()

    # Move to time 0.25 (bucket 0.2 - another new bucket)
    mock_time.return_value = 0.25
    counter.record()

    # Get throughput - should have all 4 events in window
    mock_time.return_value = 0.3
    throughput = counter.get_throughput()
    # 4 events in 5 second window = 0.8/s
    assert throughput == pytest.approx(0.8, abs=0.1)


def test_sliding_window_counter_excludes_old_buckets(mocker: MockerFixture) -> None:
    mock_time = mocker.patch("pivot_tui.stats.time.monotonic", autospec=True)

    counter = stats.SlidingWindowCounter()

    # Record events at time 0
    mock_time.return_value = 0.05
    counter.record()
    counter.record()

    # Jump forward 10 seconds (beyond 5s window)
    mock_time.return_value = 10.0
    throughput = counter.get_throughput()

    # Old events should be excluded from throughput
    assert throughput == 0.0


# =============================================================================
# MessageStatsTracker Tests
# =============================================================================


def test_message_stats_tracker_initial_state() -> None:
    tracker = stats.MessageStatsTracker("test_messages")
    result = tracker.get_stats()
    assert result["name"] == "test_messages"
    assert result["messages_received"] == 0
    assert result["messages_per_second"] == 0.0


def test_message_stats_tracker_records_messages() -> None:
    tracker = stats.MessageStatsTracker("test_messages")
    tracker.record_message()
    tracker.record_message()
    tracker.record_message()
    result = tracker.get_stats()
    assert result["messages_received"] == 3


def test_message_stats_tracker_thread_safety() -> None:
    tracker = stats.MessageStatsTracker("test_messages")
    errors = list[Exception]()

    def writer() -> None:
        try:
            for _ in range(100):
                tracker.record_message()
        except Exception as e:
            errors.append(e)

    def reader() -> None:
        try:
            for _ in range(100):
                tracker.get_stats()
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread safety errors: {errors}"
    # Both writers did 100 records each
    result = tracker.get_stats()
    assert result["messages_received"] == 200


# =============================================================================
# get_memory_mb Tests
# =============================================================================


def test_get_memory_mb_returns_positive_value() -> None:
    result = stats.get_memory_mb()
    # On systems with resource module (Unix), should return positive value
    # On systems without it (Windows), returns None
    if result is not None:
        assert result > 0, "Memory usage should be positive"
        # Sanity check: between 1MB and 100GB
        assert 1 < result < 100_000, "Memory value should be reasonable"


def test_get_memory_mb_returns_float_type() -> None:
    result = stats.get_memory_mb()
    assert result is None or isinstance(result, float)


# =============================================================================
# TypedDict Structure Tests
# =============================================================================


def test_message_stats_has_required_keys() -> None:
    tracker = stats.MessageStatsTracker("test")
    result = tracker.get_stats()
    assert "name" in result
    assert "messages_received" in result
    assert "messages_per_second" in result


def test_debug_stats_type_structure() -> None:
    # Verify the TypedDict can be constructed with expected structure
    debug_stats: stats.DebugStats = {
        "tui_messages": {
            "name": "tui",
            "messages_received": 10,
            "messages_per_second": 2.0,
        },
        "active_workers": 2,
        "memory_mb": 123.4,
        "uptime_seconds": 60.0,
    }
    assert debug_stats["tui_messages"]["name"] == "tui"
