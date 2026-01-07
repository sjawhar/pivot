"""Project root detection and path resolution.

Finds project root by locating .pivot or .git directories, and provides
utilities for resolving paths relative to the project root.
"""

import logging
import os
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


def normalize_path(path: str) -> pathlib.Path:
    """Make path absolute from project root, preserving symlinks (unlike resolve())."""
    p = pathlib.Path(path)
    abs_path = p.absolute() if p.is_absolute() else (get_project_root() / p).absolute()
    # Collapse .. components without following symlinks
    return pathlib.Path(os.path.normpath(abs_path))


def contains_symlink_in_path(path: pathlib.Path, base: pathlib.Path) -> bool:
    """Check if any component from base to path is a symlink.

    Example: If /project/data is a symlink, and path is /project/data/file.csv,
    returns True because 'data' component is a symlink.

    Args:
        path: Path to check for symlink components
        base: Base path to stop checking at

    Returns:
        True if any component in the path is a symlink
    """
    current = path.absolute()
    base_abs = base.absolute()

    while current != base_abs:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    return False


def resolve_path_for_comparison(path: str, context: str) -> pathlib.Path:
    """Resolve path for overlap comparison, falling back to normalized for missing stage outputs."""
    try:
        return resolve_path(path)
    except PermissionError as e:
        raise PermissionError(f"Permission denied for {context} '{path}'") from e
    except RuntimeError as e:
        raise RuntimeError(f"Circular symlink in {context} '{path}'") from e
    except FileNotFoundError:
        if "stage output" in context.lower():
            return normalize_path(path)
        raise
    except OSError as e:
        raise OSError(f"Filesystem error for {context} '{path}': {e}") from e
