from __future__ import annotations

import contextlib
import enum
import fcntl
import hashlib
import os
import pathlib
import time
from typing import TYPE_CHECKING, Protocol, TypedDict

from .. import path_utils

if TYPE_CHECKING:
    import io
    from collections.abc import Callable

    from .. import outputs


class LockMode(enum.IntEnum):
    READ = 0
    WRITE = 1


class LockRequest(TypedDict):
    key: str  # Canonical absolute path, trailing slash for directories
    mode: LockMode


def _add_parent_read_locks(
    key_to_mode: dict[str, LockMode],
    key: str,
    project_root: pathlib.Path,
) -> None:
    path_key = key.rstrip("/")
    parent = pathlib.Path(path_key).parent
    project_root_path = project_root

    while True:
        if project_root_path == parent:
            parent_key = parent.as_posix() + "/"
            if key_to_mode.get(parent_key) != LockMode.WRITE:
                key_to_mode[parent_key] = LockMode.READ
            break

        if project_root_path not in parent.parents:
            break

        parent_key = parent.as_posix() + "/"
        if key_to_mode.get(parent_key) != LockMode.WRITE:
            key_to_mode[parent_key] = LockMode.READ
        parent = parent.parent


def expand_lock_requests(
    deps: list[str],
    outs: list[outputs.ExpandedOut],
    project_root: pathlib.Path,
) -> list[LockRequest]:
    key_to_mode = dict[str, LockMode]()

    for dep in deps:
        key = path_utils.canonicalize_artifact_path(dep, project_root)
        if key_to_mode.get(key) != LockMode.WRITE:
            key_to_mode[key] = LockMode.READ

    for out in outs:
        key = path_utils.canonicalize_artifact_path(out.path, project_root)
        key_to_mode[key] = LockMode.WRITE

    for key in list(key_to_mode.keys()):
        _add_parent_read_locks(key_to_mode, key, project_root)

    requests = list[LockRequest]()
    for key in sorted(key_to_mode.keys()):
        requests.append(LockRequest(key=key, mode=key_to_mode[key]))
    return requests


class ArtifactLockService(Protocol):
    def acquire_many(
        self,
        requests: list[LockRequest],
        status_callback: Callable[[str, LockMode, float], None] | None = None,
    ) -> LockHandle: ...


class _HeldLock(TypedDict):
    key: str
    mode: LockMode
    file: io.TextIOWrapper


class LockHandle:
    def __init__(self, held: list[_HeldLock]) -> None:
        self._held: list[_HeldLock] = held

    def __enter__(self) -> LockHandle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.release()

    def release(self) -> None:
        try:
            for entry in reversed(self._held):
                with contextlib.suppress(OSError, ValueError):
                    fcntl.flock(entry["file"].fileno(), fcntl.LOCK_UN)
                with contextlib.suppress(OSError, ValueError):
                    entry["file"].close()
        finally:
            self._held.clear()


class LocalFlockLockService:
    def __init__(self, state_dir: pathlib.Path, retry_interval: float = 0.5) -> None:
        self._state_dir: pathlib.Path = state_dir
        self._retry_interval: float = retry_interval

    def acquire_many(
        self,
        requests: list[LockRequest],
        status_callback: Callable[[str, LockMode, float], None] | None = None,
    ) -> LockHandle:
        lock_dir = self._state_dir / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)

        key_to_mode = dict[str, LockMode]()
        for request in requests:
            if request["mode"] == LockMode.WRITE:
                key_to_mode[request["key"]] = LockMode.WRITE
            elif key_to_mode.get(request["key"]) != LockMode.WRITE:
                key_to_mode[request["key"]] = LockMode.READ

        held = list[_HeldLock]()
        try:
            for key in sorted(key_to_mode.keys()):
                mode = key_to_mode[key]
                lock_path = lock_dir / hashlib.sha256(key.encode()).hexdigest()
                lock_file = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115
                lock_flag = fcntl.LOCK_SH if mode == LockMode.READ else fcntl.LOCK_EX
                start = time.monotonic()

                while True:
                    try:
                        fcntl.flock(lock_file.fileno(), lock_flag | fcntl.LOCK_NB)
                        break
                    except InterruptedError:
                        continue  # EINTR — retry immediately without sleeping
                    except BlockingIOError:
                        elapsed = time.monotonic() - start
                        if status_callback is not None:
                            status_callback(key, mode, elapsed)
                        time.sleep(self._retry_interval)

                if mode == LockMode.WRITE:
                    _ = lock_file.seek(0)
                    _ = lock_file.truncate()
                    _ = lock_file.write(f"{key}\n{os.getpid()}\n{time.time()}\n")
                    lock_file.flush()
                held.append(_HeldLock(key=key, mode=mode, file=lock_file))
        except BaseException:
            # Release any already-acquired locks before re-raising
            LockHandle(held).release()
            raise

        return LockHandle(held)
