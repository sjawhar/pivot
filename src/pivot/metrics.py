from __future__ import annotations

import contextlib
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

# Maximum entries to prevent unbounded memory growth
MAX_ENTRIES = 100_000

_enabled = os.environ.get("PIVOT_METRICS", "").lower() in ("1", "true", "yes")

# Store durations directly by name for simplicity
_durations: dict[str, list[float]] = {}


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
    if name not in _durations:
        _durations[name] = []
    _durations[name].append(duration_ms)

    # Prevent unbounded growth - trim oldest entries if over limit
    total = sum(len(ds) for ds in _durations.values())
    if total > MAX_ENTRIES:
        # Remove oldest half of entries from each metric
        for metric_name in _durations:
            ds = _durations[metric_name]
            if len(ds) > 1:
                _durations[metric_name] = ds[len(ds) // 2 :]


def summary() -> dict[str, dict[str, float]]:
    """Summarize metrics by name: count, total_ms, avg_ms, min_ms, max_ms."""
    result = dict[str, dict[str, float]]()
    for name, durations in sorted(_durations.items()):
        if not durations:
            continue
        result[name] = {
            "count": float(len(durations)),
            "total_ms": sum(durations),
            "avg_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
        }
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
