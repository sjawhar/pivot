from __future__ import annotations

import sys
import threading
import time
from collections import deque
from typing import TypedDict


class MessageStats(TypedDict):
    """Statistics for message throughput."""

    name: str
    messages_received: int
    messages_per_second: float


class DebugStats(TypedDict):
    """Full debug statistics for TUI display."""

    tui_messages: MessageStats
    active_workers: int
    memory_mb: float | None
    uptime_seconds: float


# Sliding window configuration
_THROUGHPUT_WINDOW_SECS = 5.0
_BUCKET_COUNT = 50
_BUCKET_DURATION_SECS = _THROUGHPUT_WINDOW_SECS / _BUCKET_COUNT  # 0.1s per bucket


class SlidingWindowCounter:
    """O(1) throughput tracking with bucketed counts.

    Uses 50 buckets of 100ms each for a 5-second sliding window.
    Both record() and get_throughput() are O(1) amortized.
    """

    _buckets: deque[tuple[float, int]]  # (bucket_start_time, count)
    _current_bucket_start: float
    _current_bucket_count: int

    def __init__(self) -> None:
        self._buckets = deque(maxlen=_BUCKET_COUNT)
        self._current_bucket_start = 0.0
        self._current_bucket_count = 0

    def record(self) -> None:
        """Record a single event. O(1) amortized."""
        now = time.monotonic()
        bucket_start = now - (now % _BUCKET_DURATION_SECS)

        if bucket_start != self._current_bucket_start:
            # Flush current bucket if non-empty
            if self._current_bucket_count > 0:
                self._buckets.append((self._current_bucket_start, self._current_bucket_count))
            self._current_bucket_start = bucket_start
            self._current_bucket_count = 0

        self._current_bucket_count += 1

    def get_throughput(self) -> float:
        """Calculate events per second over the sliding window. O(bucket_count)."""
        now = time.monotonic()
        cutoff = now - _THROUGHPUT_WINDOW_SECS

        # Sum counts from recent buckets
        total = sum(count for ts, count in self._buckets if ts >= cutoff)
        # Include current bucket
        if self._current_bucket_start >= cutoff:
            total += self._current_bucket_count

        return total / _THROUGHPUT_WINDOW_SECS


class MessageStatsTracker:
    """Thread-safe tracker for message throughput statistics.

    Designed to be called from message handler (record_message)
    and debug panel updater (get_stats). All mutable state protected by lock.
    """

    _name: str
    _lock: threading.Lock
    _messages_received: int
    _throughput: SlidingWindowCounter

    def __init__(self, name: str) -> None:
        self._name = name
        self._lock = threading.Lock()
        self._messages_received = 0
        self._throughput = SlidingWindowCounter()

    def record_message(self) -> None:
        """Record that a message was received."""
        with self._lock:
            self._messages_received += 1
            self._throughput.record()

    def get_stats(self) -> MessageStats:
        """Get current statistics snapshot."""
        with self._lock:
            return MessageStats(
                name=self._name,
                messages_received=self._messages_received,
                messages_per_second=self._throughput.get_throughput(),
            )


def get_memory_mb() -> float | None:
    """Get peak RSS in MB, handling macOS vs Linux units."""
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024 * 1024)  # macOS: bytes
        return rss / 1024  # Linux: KB
    except Exception:
        return None
