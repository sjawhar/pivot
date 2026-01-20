from __future__ import annotations

import contextlib
import fnmatch
import logging
import pathlib
import threading
from typing import TYPE_CHECKING

from pivot import project, registry

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from watchfiles import Change

    from pivot.ignore import IgnoreFilter

logger = logging.getLogger(__name__)


class OutputFilter:
    """Combines output snapshot with execution state for atomic filtering.

    This enables intermediate file detection: files that are both outputs of one stage
    and inputs to another. When such files are modified externally (not by Pivot),
    we want to detect the change and trigger downstream stages.

    The approach is simple:
    - During execution: filter outputs (prevents infinite loops from Pivot's writes)
    - Not executing: DON'T filter outputs (external changes detected)

    This is simpler and more robust than time-window based approaches because:
    - No filesystem timestamp resolution issues
    - No clock skew problems on network filesystems
    - Works correctly for deleted files (Pivot clears outputs before running)
    - No race conditions between execution start and actual stage execution

    Thread-safe: methods can be called from coordinator thread while filter runs in watcher thread.
    The outputs frozenset is immutable, enabling lock-free reads during path matching.
    """

    _lock: threading.Lock
    _outputs: frozenset[pathlib.Path]
    _executing: bool

    def __init__(self, stages: list[str]) -> None:
        self._lock = threading.Lock()
        self._outputs = build_outputs_to_filter(stages)
        self._executing = False

    def update_outputs(self, stages: list[str]) -> None:
        """Atomically update output snapshot after registry reload."""
        new_outputs = build_outputs_to_filter(stages)
        with self._lock:
            self._outputs = new_outputs

    def start_execution(self) -> None:
        """Mark the start of execution."""
        with self._lock:
            self._executing = True

    def end_execution(self) -> None:
        """Mark the end of execution."""
        with self._lock:
            self._executing = False

    @contextlib.contextmanager
    def executing(self) -> Generator[None]:
        """Context manager for marking execution scope.

        Usage:
            with output_filter.executing():
                # outputs are filtered during this block
                results = executor.run(...)
        """
        self.start_execution()
        try:
            yield
        finally:
            self.end_execution()

    def should_filter(self, resolved_path: pathlib.Path) -> bool:
        """Check if path should be filtered (is output during execution).

        Returns True (filter) if:
        - Currently executing AND path matches an output

        Returns False (don't filter) if:
        - Not executing (external changes should be detected)
        - Path doesn't match any output
        """
        with self._lock:
            if not self._executing:
                return False
            outputs = self._outputs

        # Check outside lock - outputs is immutable frozenset
        return any(resolved_path == out or out in resolved_path.parents for out in outputs)


def collect_watch_paths(stages: list[str]) -> list[pathlib.Path]:
    """Collect paths: project root + dependency directories for specified stages."""
    root = project.get_project_root()
    paths: set[pathlib.Path] = {root}
    for name in stages:
        try:
            info = registry.REGISTRY.get(name)
        except KeyError:
            logger.warning(f"Stage '{name}' not found in registry, skipping")
            continue
        for dep in info["deps_paths"]:
            dep_path = project.try_resolve_path(dep)
            if dep_path is not None and dep_path.exists():
                paths.add(dep_path.parent if dep_path.is_file() else dep_path)
    return list(paths)


def get_output_paths_for_stages(stages: list[str]) -> set[str]:
    """Get output paths for specific stages only."""
    result: set[str] = set()
    for name in stages:
        try:
            info = registry.REGISTRY.get(name)
        except KeyError:
            logger.warning(f"Stage '{name}' not found in registry, skipping")
            continue
        for out_path in info["outs_paths"]:
            result.add(str(out_path))
    return result


def build_outputs_to_filter(stages_to_run: list[str]) -> frozenset[pathlib.Path]:
    """Build immutable snapshot of output paths for filtering.

    Returns a frozenset for thread-safe atomic swaps - the coordinator can replace
    the snapshot while the watcher reads the previous one without race conditions.
    """
    outputs = set[pathlib.Path]()
    for p in get_output_paths_for_stages(stages_to_run):
        resolved = project.try_resolve_path(p)
        if resolved is not None:
            outputs.add(resolved)
    return frozenset(outputs)


def create_watch_filter(
    watch_globs: list[str] | None = None,
    ignore_filter: IgnoreFilter | None = None,
    output_filter: OutputFilter | None = None,
) -> Callable[[Change, str], bool]:
    """Create filter for watch mode file events.

    Filters are applied in the watcher thread to prevent queue flooding:
    - Static patterns: bytecode, .pivotignore (no shared state)
    - Execution-time: outputs filtered during execution (prevents self-flooding)

    The output_filter combines output paths with execution state. It atomically
    tracks both the outputs snapshot (for registry reload) and execution state
    (to allow external modifications when not executing).
    """

    def watch_filter(change: Change, path: str) -> bool:
        _ = change

        # Static filters: no shared state, safe to run in watcher thread
        if ignore_filter is not None and ignore_filter.is_ignored(path):
            return False
        # Fallback: filter bytecode when no ignore_filter provided
        if ignore_filter is None and (path.endswith((".pyc", ".pyo")) or "__pycache__" in path):
            return False

        # Resolve incoming path for consistent comparison
        resolved_path = project.try_resolve_path(path)
        if resolved_path is None:
            return True  # Can't resolve, don't filter

        # Execution-time output filtering: prevents self-flooding from Pivot's own writes
        if output_filter is not None and output_filter.should_filter(resolved_path):
            return False

        # Apply glob filters if specified
        if watch_globs:
            filename = resolved_path.name
            original_path = path
            return any(
                fnmatch.fnmatch(filename, glob) or fnmatch.fnmatch(original_path, glob)
                for glob in watch_globs
            )

        return True

    return watch_filter
