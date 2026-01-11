from __future__ import annotations

import contextlib
import math
import os
import time
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Generator

# Not thread-safe by design; each worker process has its own copy (process isolation).

MAX_ENTRIES = 100_000

_enabled = os.environ.get("PIVOT_METRICS", "").lower() in ("1", "true", "yes")
_durations: dict[str, list[float]] = {}


class MetricSummary(TypedDict):
    """Summary statistics for a single metric."""

    count: int
    total_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float


def enable() -> None:
    """Enable metrics collection."""
    global _enabled
    _enabled = True


def clear() -> None:
    """Clear all collected metrics."""
    _durations.clear()


def get_entries() -> list[tuple[str, float]]:
    """Get raw entries for cross-process transfer.

    Returns list of (name, duration_ms) tuples that can be serialized
    and returned from worker processes.
    """
    return [(name, d) for name, ds in _durations.items() for d in ds]


def add_entries(entries: list[tuple[str, float]]) -> None:
    """Add entries from another process (used by main process to aggregate)."""
    for name, duration_ms in entries:
        _add(name, duration_ms)


def _add(name: str, duration_ms: float) -> None:
    """Internal: add a single metric entry."""
    # Skip invalid values that would poison summary statistics
    if math.isnan(duration_ms) or math.isinf(duration_ms):
        return

    # Prevent unbounded growth - trim before adding if at limit.
    # The sum() is O(n) but we intentionally avoid tracking a counter since metrics
    # are rarely used, trimming is even rarer, and the added complexity isn't worth it.
    if sum(len(ds) for ds in _durations.values()) >= MAX_ENTRIES:
        for metric_name, ds in _durations.items():
            _durations[metric_name] = ds[len(ds) // 2 :]

    _durations.setdefault(name, []).append(duration_ms)


def summary() -> dict[str, MetricSummary]:
    """Summarize metrics by name: count, total_ms, avg_ms, min_ms, max_ms."""
    result = dict[str, MetricSummary]()
    for name, durations in sorted(_durations.items()):
        if not durations:
            continue
        total = sum(durations)
        result[name] = MetricSummary(
            count=len(durations),
            total_ms=total,
            avg_ms=total / len(durations),
            min_ms=min(durations),
            max_ms=max(durations),
        )
    return result


@contextlib.contextmanager
def timed(name: str) -> Generator[None]:
    """Context manager to time a block of code.

    Usage:
        with metrics.timed("cache.hash_file"):
            ...

    Metrics are only collected when enabled via PIVOT_METRICS=1 or enable().
    """
    if not _enabled:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        _add(name, duration_ms)
