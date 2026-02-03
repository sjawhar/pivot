"""Path manipulation utilities."""

from __future__ import annotations


def preserve_trailing_slash(original: str, normalized: str) -> str:
    """Restore trailing slash if original had it.

    pathlib.Path operations strip trailing slashes, but DirectoryOut paths
    must preserve them. Use this after any path normalization.
    """
    if original.endswith("/") and not normalized.endswith("/"):
        return normalized + "/"
    return normalized
