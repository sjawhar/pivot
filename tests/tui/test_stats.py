from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from pivot.tui import stats

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
    mock_time = mocker.patch("pivot.tui.stats.time.monotonic", autospec=True)

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
    mock_time = mocker.patch("pivot.tui.stats.time.monotonic", autospec=True)

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
# QueueStatsTracker Tests
# =============================================================================


def test_queue_stats_tracker_initial_state() -> None:
    tracker = stats.QueueStatsTracker("test_queue")
    result = tracker.get_stats()
    assert result["name"] == "test_queue"
    assert result["messages_received"] == 0
    assert result["messages_per_second"] == 0.0
    assert result["approximate_size"] is None
    assert result["high_water_mark"] == 0


def test_queue_stats_tracker_records_messages() -> None:
    tracker = stats.QueueStatsTracker("test_queue")
    tracker.record_message()
    tracker.record_message()
    tracker.record_message()
    result = tracker.get_stats()
    assert result["messages_received"] == 3


def test_queue_stats_tracker_with_queue(mocker: MockerFixture) -> None:
    mock_queue = mocker.Mock()
    mock_queue.qsize.return_value = 5
    tracker = stats.QueueStatsTracker("test_queue", mock_queue)
    result = tracker.get_stats()
    assert result["approximate_size"] == 5
    assert result["high_water_mark"] == 5


def test_queue_stats_tracker_tracks_high_water_mark(mocker: MockerFixture) -> None:
    mock_queue = mocker.Mock()
    tracker = stats.QueueStatsTracker("test_queue", mock_queue)

    # First check with size 10
    mock_queue.qsize.return_value = 10
    result1 = tracker.get_stats()
    assert result1["high_water_mark"] == 10

    # Size drops to 3
    mock_queue.qsize.return_value = 3
    result2 = tracker.get_stats()
    assert result2["approximate_size"] == 3
    assert result2["high_water_mark"] == 10, "High water mark should stay at 10"

    # Size rises to 15
    mock_queue.qsize.return_value = 15
    result3 = tracker.get_stats()
    assert result3["high_water_mark"] == 15


def test_queue_stats_tracker_handles_qsize_not_implemented(mocker: MockerFixture) -> None:
    mock_queue = mocker.Mock()
    mock_queue.qsize.side_effect = NotImplementedError
    tracker = stats.QueueStatsTracker("test_queue", mock_queue)
    result = tracker.get_stats()
    assert result["approximate_size"] is None


def test_queue_stats_tracker_thread_safety() -> None:
    tracker = stats.QueueStatsTracker("test_queue")
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


def test_queue_stats_has_required_keys() -> None:
    tracker = stats.QueueStatsTracker("test")
    result = tracker.get_stats()
    assert "name" in result
    assert "messages_received" in result
    assert "messages_per_second" in result
    assert "approximate_size" in result
    assert "high_water_mark" in result


def test_debug_stats_type_structure() -> None:
    # Verify the TypedDict can be constructed with expected structure
    debug_stats: stats.DebugStats = {
        "tui_queue": {
            "name": "tui",
            "messages_received": 10,
            "messages_per_second": 2.0,
            "approximate_size": 5,
            "high_water_mark": 8,
        },
        "output_queue": None,
        "active_workers": 2,
        "memory_mb": 123.4,
        "uptime_seconds": 60.0,
    }
    assert debug_stats["tui_queue"]["name"] == "tui"
    assert debug_stats["output_queue"] is None
