from __future__ import annotations

import fnmatch
import logging
from typing import TYPE_CHECKING

from pivot import project, registry

if TYPE_CHECKING:
    import pathlib
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


def _matches_glob(resolved_path: pathlib.Path, original_path: str, watch_globs: list[str]) -> bool:
    """Check if path matches any of the glob patterns."""
    filename = resolved_path.name
    return any(
        fnmatch.fnmatch(filename, glob) or fnmatch.fnmatch(original_path, glob)
        for glob in watch_globs
    )


def create_watch_filter(
    watch_globs: list[str] | None = None,
) -> Callable[[Change, str], bool]:
    """Create stateless filter for watch mode.

    Filters:
    - Python bytecode (.pyc, .pyo, __pycache__)
    - Files not matching watch_globs (if specified)

    Output filtering is NOT done here - it's handled naturally by the file index
    which only maps dependencies (inputs), not outputs. Changes to output-only
    files won't match any stage. Changes to intermediate files (output of A,
    input of B) will correctly trigger B.
    """

    def watch_filter(change: Change, path: str) -> bool:
        _ = change

        # Always filter Python bytecode
        if path.endswith((".pyc", ".pyo")) or "__pycache__" in path:
            return False

        # Apply glob filters if specified
        if watch_globs:
            resolved_path = project.try_resolve_path(path)
            if resolved_path is None:
                return True  # Can't resolve, don't filter
            return _matches_glob(resolved_path, path, watch_globs)

        return True

    return watch_filter
