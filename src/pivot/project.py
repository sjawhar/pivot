"""Project root detection and path resolution.

Finds project root by locating .pivot or .git directories, and provides
utilities for resolving paths relative to the project root.
"""

import logging
import pathlib

logger = logging.getLogger(__name__)

_project_root_cache: pathlib.Path | None = None


def find_project_root() -> pathlib.Path:
    """Walk up from cwd to find .pivot or .git directory."""
    current = pathlib.Path.cwd().resolve()
    for parent in [current, *current.parents]:
        if (parent / ".pivot").exists() or (parent / ".git").exists():
            logger.debug(f"Found project root: {parent}")
            return parent

    logger.warning("No project markers (.pivot or .git) found, using current directory")
    return current


def get_project_root() -> pathlib.Path:
    """Get project root (cached after first call)."""
    global _project_root_cache
    if _project_root_cache is None:
        _project_root_cache = find_project_root()
        logger.info(f"Project root: {_project_root_cache}")
    return _project_root_cache


def resolve_path(path: str) -> pathlib.Path:
    """Resolve relative path from project root; absolute paths unchanged."""
    p = pathlib.Path(path)
    if p.is_absolute():
        return p.resolve()
    return (get_project_root() / p).resolve()
