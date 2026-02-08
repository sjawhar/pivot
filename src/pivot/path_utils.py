"""Path manipulation utilities."""

from __future__ import annotations

import os
import pathlib


def canonicalize_artifact_path(path: str, base: pathlib.Path) -> str:
    """Produce the single canonical form for an artifact path.

    Canonical form:
    - Absolute (resolved from base if relative)
    - Normalized (no .., no //, no trailing dots)
    - POSIX separators (backslashes converted to forward slashes)
    - Trailing slash preserved for directory artifacts (DirectoryOut)

    This is the ONE function that should be used to produce artifact paths
    for in-memory use (registry, DAG, engine). Lockfiles convert to/from
    project-relative at their own boundary.

    Args:
        path: Raw artifact path (relative or absolute).
        base: Base directory for resolving relative paths.

    Returns:
        Canonical absolute path string, with trailing slash preserved if input had one.
    """
    has_trailing_slash = path.endswith("/")
    # Normalize separators to POSIX before pathlib processing
    posix_path = path.replace("\\", "/")
    p = pathlib.Path(posix_path)
    abs_path = p if p.is_absolute() else base / p
    normalized = pathlib.Path(os.path.normpath(abs_path))
    result = normalized.as_posix()
    if has_trailing_slash and not result.endswith("/"):
        result += "/"
    return result


def preserve_trailing_slash(original: str, normalized: str) -> str:
    """Restore trailing slash if original had it.

    pathlib.Path operations strip trailing slashes, but DirectoryOut paths
    must preserve them. Use this after any path normalization.
    """
    if original.endswith("/") and not normalized.endswith("/"):
        return normalized + "/"
    return normalized
