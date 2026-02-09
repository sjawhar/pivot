"""Artifact lock request modeling for concurrent execution."""

from __future__ import annotations

import enum
import fcntl
import hashlib
import os
import pathlib
import time
from collections.abc import Callable
from typing import TypedDict, final

from pivot import outputs, path_utils


class LockMode(enum.IntEnum):
    """Lock strength for an artifact path."""

    READ = 0
    WRITE = 1


StatusCallback = Callable[[str, LockMode, float], None]


class LockRequest(TypedDict):
    """Lock request for an artifact path."""

    key: str
    mode: LockMode


def expand_lock_requests(
    deps: list[str],
    outs: list[outputs.ExpandedOut],
    project_root: pathlib.Path,
) -> list[LockRequest]:
    """Expand deps/outs into lock requests with ancestor directory reads."""
    key_to_mode = dict[str, LockMode]()

    for dep in deps:
        key = path_utils.canonicalize_artifact_path(dep, project_root)
        key = key.rstrip("/")
        if not key:
            key = "/"
        if key in key_to_mode and key_to_mode[key] is LockMode.WRITE:
            continue
        key_to_mode[key] = LockMode.READ

    for out in outs:
        key = path_utils.canonicalize_artifact_path(out.path, project_root)
        key = key.rstrip("/")
        if not key:
            key = "/"
        key_to_mode[key] = LockMode.WRITE

    keys = list(key_to_mode)
    for key in keys:
        base = key.rstrip("/")
        path = pathlib.Path(base)
        for parent in path.parents:
            if project_root in parent.parents:
                _add_read_lock(key_to_mode, parent)
                continue
            if parent == project_root:
                _add_read_lock(key_to_mode, parent)
                break
            break

    return [LockRequest(key=key, mode=key_to_mode[key]) for key in sorted(key_to_mode)]


def _add_read_lock(key_to_mode: dict[str, LockMode], path: pathlib.Path) -> None:
    key = path.as_posix()
    if not key.endswith("/"):
        key += "/"
    if key in key_to_mode and key_to_mode[key] is LockMode.WRITE:
        return
    key_to_mode[key] = LockMode.READ


# ---------------------------------------------------------------------------
# Runtime flock-based lock service
# ---------------------------------------------------------------------------

_RETRY_INTERVAL = 0.5


@final
class LockHandle:
    """Context manager holding acquired file descriptors for artifact locks.

    Releases locks in reverse acquisition order on exit.
    """

    __slots__ = ("_fds",)

    def __init__(self, fds: list[tuple[int, str]]) -> None:
        self._fds = fds

    def release(self) -> None:
        """Release all locks in reverse order and close fds."""
        for fd, _key in reversed(self._fds):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        self._fds = list[tuple[int, str]]()

    def __enter__(self) -> LockHandle:
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


@final
class LocalFlockLockService:
    """File-system lock service using POSIX flock for artifact coordination."""

    __slots__ = ("_lock_dir",)

    def __init__(self, lock_dir: pathlib.Path) -> None:
        lock_dir.mkdir(parents=True, exist_ok=True)
        self._lock_dir = lock_dir

    def acquire_many(
        self,
        requests: list[LockRequest],
        on_status: StatusCallback | None = None,
    ) -> LockHandle:
        """Acquire locks for *requests*, returning a `LockHandle`.

        Requests are normalized to the strongest mode per key (WRITE > READ)
        and acquired in sorted key order to avoid deadlocks.
        """
        # Normalize: strongest mode per key
        key_to_mode = dict[str, LockMode]()
        for req in requests:
            key = req["key"]
            mode = req["mode"]
            if key not in key_to_mode or mode > key_to_mode[key]:
                key_to_mode[key] = mode

        fds = list[tuple[int, str]]()
        try:
            for key in sorted(key_to_mode):
                mode = key_to_mode[key]
                fd = self._acquire_one(key, mode, on_status)
                fds.append((fd, key))
        except BaseException:
            # Release already-acquired locks on failure
            for fd, _key in reversed(fds):
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
            raise

        return LockHandle(fds)

    def _acquire_one(
        self,
        key: str,
        mode: LockMode,
        on_status: StatusCallback | None,
    ) -> int:
        filename = hashlib.sha256(key.encode()).hexdigest()
        lock_path = self._lock_dir / filename
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)

        op = fcntl.LOCK_SH if mode is LockMode.READ else fcntl.LOCK_EX
        start = time.monotonic()
        try:
            while True:
                try:
                    fcntl.flock(fd, op | fcntl.LOCK_NB)
                    return fd
                except BlockingIOError:
                    elapsed = time.monotonic() - start
                    if on_status is not None:
                        on_status(key, mode, elapsed)
                    time.sleep(_RETRY_INTERVAL)
        except BaseException:
            os.close(fd)
            raise
