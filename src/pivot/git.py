from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import dulwich.errors
import dulwich.object_store
import dulwich.objects
import dulwich.repo

from pivot import project

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def _find_git_root(start: pathlib.Path) -> pathlib.Path | None:
    """Walk up to find .git directory."""
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def read_file_from_head(rel_path: str) -> bytes | None:
    """Read file contents from HEAD commit.

    Args:
        rel_path: Path relative to project root

    Returns:
        File contents as bytes, or None if not found
    """
    proj_root = project.get_project_root()
    git_root = _find_git_root(proj_root)

    if git_root is None:
        logger.debug("No git repository found")
        return None

    try:
        repo = dulwich.repo.Repo(str(git_root))
    except dulwich.errors.NotGitRepository:
        logger.debug(f"Not a git repository: {git_root}")
        return None

    try:
        head_sha = repo.head()
    except KeyError:
        logger.debug("No HEAD commit (empty repository?)")
        return None

    commit = repo[head_sha]
    if not isinstance(commit, dulwich.objects.Commit):
        logger.debug(f"HEAD is not a commit: {type(commit)}")
        return None

    # If project root is inside git root, adjust path
    if proj_root != git_root:
        try:
            proj_prefix = proj_root.relative_to(git_root)
            full_path = str(proj_prefix / rel_path)
        except ValueError:
            full_path = rel_path
    else:
        full_path = rel_path

    try:
        _mode, sha = dulwich.object_store.tree_lookup_path(
            repo.__getitem__, commit.tree, full_path.encode()
        )
        blob = repo[sha]
        if isinstance(blob, dulwich.objects.Blob):
            return blob.data
        return None
    except KeyError:
        logger.debug(f"File not found in HEAD: {full_path}")
        return None


def read_files_from_head(rel_paths: Sequence[str]) -> dict[str, bytes]:
    """Read multiple files from HEAD commit efficiently.

    Args:
        rel_paths: Paths relative to project root

    Returns:
        Dict mapping path to contents for files that exist in HEAD
    """
    if not rel_paths:
        return {}

    proj_root = project.get_project_root()
    git_root = _find_git_root(proj_root)

    if git_root is None:
        return {}

    try:
        repo = dulwich.repo.Repo(str(git_root))
    except dulwich.errors.NotGitRepository:
        return {}

    try:
        head_sha = repo.head()
    except KeyError:
        return {}

    commit = repo[head_sha]
    if not isinstance(commit, dulwich.objects.Commit):
        return {}

    # Calculate prefix adjustment once
    try:
        proj_prefix = proj_root.relative_to(git_root) if proj_root != git_root else None
    except ValueError:
        proj_prefix = None

    result = dict[str, bytes]()
    for rel_path in rel_paths:
        full_path = str(proj_prefix / rel_path) if proj_prefix else rel_path
        try:
            _mode, sha = dulwich.object_store.tree_lookup_path(
                repo.__getitem__, commit.tree, full_path.encode()
            )
            blob = repo[sha]
            if isinstance(blob, dulwich.objects.Blob):
                result[rel_path] = blob.data
        except KeyError:
            pass

    return result
