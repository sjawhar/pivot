from __future__ import annotations

import fnmatch
import logging
import pathlib
from typing import TYPE_CHECKING

from pivot import project, registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from watchfiles import Change

logger = logging.getLogger(__name__)


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
        for dep in info["deps"]:
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


def create_watch_filter(
    stages_to_run: list[str],
    watch_globs: list[str] | None = None,
) -> Callable[[Change, str], bool]:
    """Create filter excluding outputs from stages being run (prevents infinite loops)."""
    outputs_to_filter = set[pathlib.Path]()
    for p in get_output_paths_for_stages(stages_to_run):
        resolved = project.try_resolve_path(p)
        if resolved is not None:
            outputs_to_filter.add(resolved)

    def watch_filter(change: Change, path: str) -> bool:
        _ = change

        # Always filter Python bytecode
        if path.endswith((".pyc", ".pyo")) or "__pycache__" in path:
            return False

        # Resolve incoming path for consistent comparison
        resolved_path = project.try_resolve_path(path)
        if resolved_path is None:
            return True  # Can't resolve, don't filter

        # Check if path is an output of a stage being run, or inside such an output directory
        for out in outputs_to_filter:
            if resolved_path == out or out in resolved_path.parents:
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
