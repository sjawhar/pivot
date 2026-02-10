"""Git archive fallback for reading files from non-GitHub remote repos.

Uses ``git archive --remote`` and ``git ls-remote`` via subprocess.
GitHub has **disabled** ``git archive --remote`` for public repos —
this module targets self-hosted Git servers and GitLab.
"""

from __future__ import annotations

import io
import logging
import subprocess
import tarfile

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30  # seconds


def resolve_ref_from_remote_repo(
    repo_url: str,
    ref: str,
    *,
    timeout: int = _DEFAULT_TIMEOUT,
) -> str | None:
    """Resolve a ref to a commit SHA using ``git ls-remote``.

    Returns the 40-char hex SHA, or ``None`` if the ref cannot be resolved
    (repo not found, branch missing, timeout, git not installed).
    """
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--", repo_url, ref],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            logger.debug(
                "git ls-remote failed for %s: %s",
                repo_url,
                result.stderr,
            )
            return None
        # Prefer dereferenced ^{} SHA (commit) over tag object SHA.
        # git ls-remote returns both for annotated tags:
        #   <tag-object-sha>  refs/tags/v1.0
        #   <commit-sha>      refs/tags/v1.0^{}
        match_sha: str | None = None
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            sha, ref_name = parts[0], parts[1]
            is_deref = ref_name.endswith("^{}")
            bare_ref = ref_name.removesuffix("^{}")
            if bare_ref == ref or bare_ref.endswith(f"/{ref}"):
                if is_deref:
                    return sha
                match_sha = sha
        return match_sha
    except subprocess.TimeoutExpired:
        logger.warning(
            "git ls-remote timed out for %s after %ds",
            repo_url,
            timeout,
        )
        return None
    except FileNotFoundError:
        logger.warning("git command not found")
        return None


def read_file_from_remote_repo(
    repo_url: str,
    path: str,
    rev: str,
    *,
    timeout: int = _DEFAULT_TIMEOUT,
) -> bytes | None:
    """Read a single file from a remote repo using ``git archive``.

    Returns file contents as bytes, or ``None`` on any failure
    (path not found, repo inaccessible, timeout, tar extraction error).
    """
    try:
        result = subprocess.run(
            ["git", "archive", f"--remote={repo_url}", rev, "--", path],
            capture_output=True,
            timeout=timeout,
            check=True,
        )
        with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        return f.read()
        return None
    except subprocess.CalledProcessError as e:
        logger.debug("git archive failed: %s", e.stderr)
        return None
    except subprocess.TimeoutExpired:
        logger.warning(
            "git archive timed out for %s after %ds",
            repo_url,
            timeout,
        )
        return None
    except FileNotFoundError:
        logger.warning("git command not found")
        return None
    except (tarfile.TarError, OSError) as e:
        logger.debug("Failed to extract tar: %s", e)
        return None


def list_directory_from_remote_repo(
    repo_url: str,
    path: str,
    rev: str,
    *,
    timeout: int = _DEFAULT_TIMEOUT,
) -> list[str] | None:
    """List files in a directory from a remote repo via ``git archive``.

    Returns a list of file paths (relative to repo root), or ``None``
    on failure (path not found, repo inaccessible, timeout).
    """
    try:
        result = subprocess.run(
            ["git", "archive", f"--remote={repo_url}", rev, "--", path],
            capture_output=True,
            timeout=timeout,
            check=True,
        )
        with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
            return [m.name.rsplit("/", 1)[-1] for m in tar.getmembers() if m.isfile()]
    except subprocess.CalledProcessError as e:
        logger.debug("git archive failed: %s", e.stderr)
        return None
    except subprocess.TimeoutExpired:
        logger.warning(
            "git archive timed out for %s after %ds",
            repo_url,
            timeout,
        )
        return None
    except FileNotFoundError:
        logger.warning("git command not found")
        return None
    except (tarfile.TarError, OSError) as e:
        logger.debug("Failed to extract tar: %s", e)
        return None
