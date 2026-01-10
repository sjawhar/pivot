from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, cast

import dulwich.errors
import dulwich.object_store
import dulwich.objects
import dulwich.refs
import dulwich.repo

from pivot import project

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)


class _RepoContext(NamedTuple):
    """Internal context for git operations."""

    repo: dulwich.repo.Repo
    commit: dulwich.objects.Commit
    proj_prefix: Path | None


def _find_git_root(start: Path) -> Path | None:
    """Walk up to find .git directory."""
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _open_repo() -> tuple[dulwich.repo.Repo, Path, Path] | None:
    """Open git repo and return (repo, git_root, proj_root) or None."""
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

    return repo, git_root, proj_root


def _get_proj_prefix(git_root: Path, proj_root: Path) -> Path | None:
    """Calculate project prefix relative to git root."""
    if proj_root == git_root:
        return None
    try:
        return proj_root.relative_to(git_root)
    except ValueError:
        return None


def _resolve_path(proj_prefix: Path | None, rel_path: str) -> str:
    """Resolve relative path accounting for project prefix."""
    return str(proj_prefix / rel_path) if proj_prefix else rel_path


def _read_blob(
    repo: dulwich.repo.Repo,
    commit: dulwich.objects.Commit,
    proj_prefix: Path | None,
    rel_path: str,
) -> bytes | None:
    """Read a single blob from commit."""
    full_path = _resolve_path(proj_prefix, rel_path)
    try:
        _mode, sha = dulwich.object_store.tree_lookup_path(
            repo.__getitem__, commit.tree, full_path.encode()
        )
        blob = repo[sha]
        if isinstance(blob, dulwich.objects.Blob):
            return blob.data
    except KeyError:
        logger.debug(f"File not found: {full_path}")
    return None


def _get_head_context() -> _RepoContext | None:
    """Get repo context for HEAD commit."""
    result = _open_repo()
    if result is None:
        return None

    repo, git_root, proj_root = result

    try:
        head_sha = repo.head()
    except KeyError:
        logger.debug("No HEAD commit (empty repository?)")
        return None

    commit = repo[head_sha]
    if not isinstance(commit, dulwich.objects.Commit):
        logger.debug(f"HEAD is not a commit: {type(commit)}")
        return None

    proj_prefix = _get_proj_prefix(git_root, proj_root)
    return _RepoContext(repo=repo, commit=commit, proj_prefix=proj_prefix)


def is_git_repo_with_head() -> bool:
    """Check if we're in a git repo with a valid HEAD commit."""
    return _get_head_context() is not None


def _get_revision_context(rev: str) -> _RepoContext | None:
    """Get repo context for a specific revision."""
    result = _open_repo()
    if result is None:
        return None

    repo, git_root, proj_root = result

    commit_sha_str = _resolve_revision_with_repo(repo, rev)
    if commit_sha_str is None:
        return None

    commit_sha = commit_sha_str.encode()
    commit = repo[commit_sha]
    if not isinstance(commit, dulwich.objects.Commit):
        return None

    proj_prefix = _get_proj_prefix(git_root, proj_root)
    return _RepoContext(repo=repo, commit=commit, proj_prefix=proj_prefix)


def _dereference_to_commit(
    repo: dulwich.repo.Repo, obj: dulwich.objects.ShaFile, sha_hex: str
) -> str | None:
    """Dereference tags to get commit SHA. Returns 40-char hex string or None."""
    while isinstance(obj, dulwich.objects.Tag):
        # Tag.object[1] is 20 raw bytes, need to convert to hex for return value
        # But repo[] accepts both raw bytes and hex bytes, so use raw for lookup
        target_sha_raw = obj.object[1]
        sha_hex = target_sha_raw.hex()
        obj = repo[target_sha_raw]

    if isinstance(obj, dulwich.objects.Commit):
        return sha_hex
    return None


def _resolve_revision_with_repo(repo: dulwich.repo.Repo, rev: str) -> str | None:
    """Resolve revision using an already-open repo."""
    rev_bytes = rev.encode()

    # Try refs first (branches and tags) - this is the common case
    refs_to_try = [
        rev_bytes,
        b"refs/heads/" + rev_bytes,
        b"refs/tags/" + rev_bytes,
        b"refs/remotes/origin/" + rev_bytes,
    ]

    for ref in refs_to_try:
        try:
            # dulwich refs return 40-char hex string as bytes
            sha_bytes = repo.refs[cast("dulwich.refs.Ref", ref)]
            sha_hex = sha_bytes.decode()
            obj = repo[sha_bytes]
            result = _dereference_to_commit(repo, obj, sha_hex)
            if result is not None:
                return result
        except KeyError:
            continue

    # Try as hex SHA (full or short)
    if len(rev) >= 4 and all(c in "0123456789abcdefABCDEF" for c in rev):
        rev_lower = rev.lower()
        try:
            # object_store iteration yields 40-char hex string as bytes
            for sha_bytes in repo.object_store:
                sha_hex = sha_bytes.decode()
                if sha_hex.lower().startswith(rev_lower):
                    obj = repo[sha_bytes]
                    result = _dereference_to_commit(repo, obj, sha_hex)
                    if result is not None:
                        return result
        except (KeyError, ValueError):
            pass

    logger.debug(f"Could not resolve revision: {rev}")
    return None


def read_file_from_head(rel_path: str) -> bytes | None:
    """Read file contents from HEAD commit."""
    ctx = _get_head_context()
    if ctx is None:
        return None
    return _read_blob(ctx.repo, ctx.commit, ctx.proj_prefix, rel_path)


def read_files_from_head(rel_paths: Sequence[str]) -> dict[str, bytes]:
    """Read multiple files from HEAD commit efficiently."""
    if not rel_paths:
        return {}

    ctx = _get_head_context()
    if ctx is None:
        return {}

    result = dict[str, bytes]()
    for rel_path in rel_paths:
        content = _read_blob(ctx.repo, ctx.commit, ctx.proj_prefix, rel_path)
        if content is not None:
            result[rel_path] = content
    return result


def resolve_revision(rev: str) -> str | None:
    """Resolve git revision (SHA, branch, tag) to commit SHA.

    Returns commit SHA as 40-char hex string, or None if invalid/not found.
    """
    result = _open_repo()
    if result is None:
        return None

    repo, _git_root, _proj_root = result
    return _resolve_revision_with_repo(repo, rev)


def read_file_from_revision(rel_path: str, rev: str) -> bytes | None:
    """Read file contents from a specific git revision."""
    ctx = _get_revision_context(rev)
    if ctx is None:
        return None
    return _read_blob(ctx.repo, ctx.commit, ctx.proj_prefix, rel_path)


def read_files_from_revision(rel_paths: Sequence[str], rev: str) -> dict[str, bytes]:
    """Read multiple files from a specific git revision efficiently."""
    if not rel_paths:
        return {}

    ctx = _get_revision_context(rev)
    if ctx is None:
        return {}

    result = dict[str, bytes]()
    for rel_path in rel_paths:
        content = _read_blob(ctx.repo, ctx.commit, ctx.proj_prefix, rel_path)
        if content is not None:
            result[rel_path] = content
    return result
