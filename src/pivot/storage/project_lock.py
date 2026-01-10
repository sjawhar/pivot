from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import filelock

from pivot import project

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator

logger = logging.getLogger(__name__)

_PENDING_LOCK_NAME = "pending.lock"


def _get_lock_path() -> pathlib.Path:
    """Get path to pending state coordination lock file."""
    return project.get_project_root() / ".pivot" / _PENDING_LOCK_NAME


def _create_lock(timeout: float = -1) -> filelock.BaseFileLock:
    """Create a FileLock with standard setup (creates parent dir if needed)."""
    lock_path = _get_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    return filelock.FileLock(lock_path, timeout=timeout)


@contextlib.contextmanager
def pending_state_lock(timeout: float = -1) -> Generator[filelock.BaseFileLock]:
    """Context manager for coordinating --no-commit execution and commit operations.

    Yields the lock object for callers that need it (e.g., to check lock state).
    """
    lock = _create_lock(timeout=timeout)
    logger.debug(f"Acquiring pending state lock: {lock.lock_file}")
    with lock:
        logger.debug("Pending state lock acquired")
        yield lock
    logger.debug("Released pending state lock")


def acquire_pending_state_lock(timeout: float = -1) -> filelock.BaseFileLock:
    """Acquire lock for commit/discard operations (blocking with optional timeout).

    Args:
        timeout: Seconds to wait. -1 means wait forever, 0 means non-blocking.

    Returns:
        BaseFileLock (caller must release).

    Raises:
        filelock.Timeout: If timeout >= 0 and lock not acquired in time.
    """
    lock = _create_lock(timeout=timeout)
    logger.debug(f"Waiting for pending state lock: {lock.lock_file}")
    lock.acquire()
    logger.debug("Pending state lock acquired")
    return lock
