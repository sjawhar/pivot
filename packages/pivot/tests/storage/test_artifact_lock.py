from __future__ import annotations

import enum
import importlib
import os
import pathlib
import threading
import time
from typing import Any, Protocol, cast


class _LoadersModule(Protocol):
    def JSON(self) -> object: ...  # noqa: N802


class _OutputsModule(Protocol):
    def Out(self, path: str, loader: object) -> object: ...  # noqa: N802

    def DirectoryOut(self, path: str, loader: object) -> object: ...  # noqa: N802


class _LockMode(enum.IntEnum):
    READ = 0
    WRITE = 1


class _ArtifactLockModule(Protocol):
    LockMode: type[_LockMode]

    def expand_lock_requests(
        self,
        deps: list[str],
        outs: list[object],
        project_root: pathlib.Path,
    ) -> list[dict[str, object]]: ...


loaders = cast("_LoadersModule", cast("object", importlib.import_module("pivot.loaders")))
outputs = cast("_OutputsModule", cast("object", importlib.import_module("pivot.outputs")))
artifact_lock = cast(
    "_ArtifactLockModule",
    cast("object", importlib.import_module("pivot.storage.artifact_lock")),
)


def _helper_lock_map(requests: list[dict[str, object]]) -> dict[str, object]:
    return {cast("str", request["key"]): request["mode"] for request in requests}


def test_expand_lock_requests_empty() -> None:
    project_root = pathlib.Path("/project")

    result = artifact_lock.expand_lock_requests(list[str](), list[object](), project_root)

    assert result == list[dict[str, object]]()


def test_expand_lock_requests_dep_read_and_ancestors() -> None:
    project_root = pathlib.Path("/project")

    result = artifact_lock.expand_lock_requests(["data/input.csv"], list[object](), project_root)

    lock_map = _helper_lock_map(result)
    assert lock_map["/project/data/input.csv"] is artifact_lock.LockMode.READ
    assert lock_map["/project/data/"] is artifact_lock.LockMode.READ
    assert lock_map["/project/"] is artifact_lock.LockMode.READ


def test_expand_lock_requests_out_write_and_ancestor() -> None:
    project_root = pathlib.Path("/project")
    out = outputs.Out("output.csv", loaders.JSON())

    result = artifact_lock.expand_lock_requests(list[str](), [out], project_root)

    lock_map = _helper_lock_map(result)
    assert lock_map["/project/output.csv"] is artifact_lock.LockMode.WRITE
    assert lock_map["/project/"] is artifact_lock.LockMode.READ


def test_write_dominates_read() -> None:
    project_root = pathlib.Path("/project")
    out = outputs.Out("output.csv", loaders.JSON())

    result = artifact_lock.expand_lock_requests(["output.csv"], [out], project_root)

    lock_map = _helper_lock_map(result)
    assert lock_map["/project/output.csv"] is artifact_lock.LockMode.WRITE


def test_directory_out_key_normalized() -> None:
    project_root = pathlib.Path("/project")
    out = outputs.DirectoryOut("dir/", loaders.JSON())

    result = artifact_lock.expand_lock_requests(list[str](), [out], project_root)

    lock_map = _helper_lock_map(result)
    assert lock_map["/project/dir"] is artifact_lock.LockMode.WRITE
    assert "/project/dir/" not in lock_map


def test_expand_directory_and_file_same_key() -> None:
    project_root = pathlib.Path("/project")
    out_with_slash = outputs.Out("data/", loaders.JSON())
    out_without_slash = outputs.Out("data", loaders.JSON())

    result_with_slash = artifact_lock.expand_lock_requests(
        list[str](), [out_with_slash], project_root
    )
    result_without_slash = artifact_lock.expand_lock_requests(
        list[str](), [out_without_slash], project_root
    )

    lock_map_with_slash = _helper_lock_map(result_with_slash)
    lock_map_without_slash = _helper_lock_map(result_without_slash)

    assert "/project/data" in lock_map_with_slash
    assert "/project/data" in lock_map_without_slash
    assert lock_map_with_slash["/project/data"] is artifact_lock.LockMode.WRITE
    assert lock_map_without_slash["/project/data"] is artifact_lock.LockMode.WRITE


def test_deterministic_sort() -> None:
    project_root = pathlib.Path("/project")

    result = artifact_lock.expand_lock_requests(["b.txt", "a.txt"], list[object](), project_root)

    keys = [request["key"] for request in result]
    assert keys == ["/project/", "/project/a.txt", "/project/b.txt"]


def test_stop_at_project_root() -> None:
    project_root = pathlib.Path("/project")

    result = artifact_lock.expand_lock_requests(
        ["/project/data/input.csv"], list[object](), project_root
    )

    lock_map = _helper_lock_map(result)
    assert "/" not in lock_map


# ---------------------------------------------------------------------------
# Flock-based LocalFlockLockService tests
# ---------------------------------------------------------------------------


_flock_mod: Any = importlib.import_module("pivot.storage.artifact_lock")
_FlockLockMode: type[_LockMode] = _flock_mod.LockMode
_FlockLockService: Any = _flock_mod.LocalFlockLockService
_FlockLockHandle: Any = _flock_mod.LockHandle


def _helper_make_request(key: str, mode: _LockMode) -> dict[str, object]:
    return {"key": key, "mode": mode}


def test_flock_single_acquire_release(tmp_path: pathlib.Path) -> None:
    svc = _FlockLockService(tmp_path / "locks")
    requests = [_helper_make_request("data/input.csv", _FlockLockMode.READ)]

    handle = svc.acquire_many(requests)
    assert isinstance(handle, _FlockLockHandle)
    handle.release()


def test_flock_exclusive_serialize(tmp_path: pathlib.Path) -> None:
    svc = _FlockLockService(tmp_path / "locks")
    key = "shared/resource"
    order = list[str]()
    barrier = threading.Barrier(2, timeout=5)

    def _worker(name: str) -> None:
        barrier.wait()
        with svc.acquire_many([_helper_make_request(key, _FlockLockMode.WRITE)]):
            order.append(f"{name}_start")
            time.sleep(0.1)
            order.append(f"{name}_end")

    t1 = threading.Thread(target=_worker, args=("a",))
    t2 = threading.Thread(target=_worker, args=("b",))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert len(order) == 4, f"Expected 4 events, got {order}"
    assert order[0].endswith("_start")
    assert order[1].endswith("_end")
    first = order[0].split("_")[0]
    second = order[2].split("_")[0]
    assert first != second, "Both threads must have different names"
    assert order[2].endswith("_start")
    assert order[3].endswith("_end")


def test_flock_shared_concurrent(tmp_path: pathlib.Path) -> None:
    svc = _FlockLockService(tmp_path / "locks")
    key = "shared/data"
    held = list[bool]()
    barrier = threading.Barrier(2, timeout=5)

    def _reader() -> None:
        with svc.acquire_many([_helper_make_request(key, _FlockLockMode.READ)]):
            barrier.wait()
            held.append(True)

    t1 = threading.Thread(target=_reader)
    t2 = threading.Thread(target=_reader)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(held) == 2, "Both shared readers should hold locks concurrently"


def test_flock_write_blocks_shared(tmp_path: pathlib.Path) -> None:
    svc = _FlockLockService(tmp_path / "locks")
    key = "exclusive/resource"
    writer_entered = threading.Event()
    reader_acquired = threading.Event()

    def _writer() -> None:
        with svc.acquire_many([_helper_make_request(key, _FlockLockMode.WRITE)]):
            writer_entered.set()
            time.sleep(0.4)

    def _reader() -> None:
        writer_entered.wait(timeout=5)
        time.sleep(0.05)
        with svc.acquire_many([_helper_make_request(key, _FlockLockMode.READ)]):
            reader_acquired.set()

    tw = threading.Thread(target=_writer)
    tr = threading.Thread(target=_reader)
    tw.start()
    tr.start()
    tw.join(timeout=10)
    tr.join(timeout=10)

    assert reader_acquired.is_set(), "Reader should eventually acquire after writer releases"


def test_flock_lock_files_created(tmp_path: pathlib.Path) -> None:
    lock_dir = tmp_path / "locks"
    svc = _FlockLockService(lock_dir)
    requests = [
        _helper_make_request("alpha", _FlockLockMode.READ),
        _helper_make_request("beta", _FlockLockMode.WRITE),
    ]

    with svc.acquire_many(requests):
        lock_files = list(lock_dir.iterdir())
        assert len(lock_files) == 2, f"Expected 2 lock files, got {lock_files}"


def test_flock_status_callback_invoked(tmp_path: pathlib.Path) -> None:
    svc = _FlockLockService(tmp_path / "locks")
    key = "contended/resource"
    callback_calls = list[tuple[str, object, float]]()
    writer_holding = threading.Event()
    callback_fired = threading.Event()

    def _on_status(k: str, mode: object, elapsed: float) -> None:
        callback_calls.append((k, mode, elapsed))
        callback_fired.set()

    def _holder() -> None:
        with svc.acquire_many([_helper_make_request(key, _FlockLockMode.WRITE)]):
            writer_holding.set()
            callback_fired.wait(timeout=5)
            time.sleep(0.1)

    holder = threading.Thread(target=_holder)
    holder.start()
    writer_holding.wait(timeout=5)

    with svc.acquire_many([_helper_make_request(key, _FlockLockMode.WRITE)], on_status=_on_status):
        pass

    holder.join(timeout=10)

    assert len(callback_calls) >= 1, "Callback should fire at least once during contention"
    assert callback_calls[0][0] == key
    assert callback_calls[0][1] is _FlockLockMode.WRITE
    assert callback_calls[0][2] >= 0.0


def test_flock_crash_recovery_fd_close(tmp_path: pathlib.Path) -> None:
    svc = _FlockLockService(tmp_path / "locks")
    key = "crash/test"

    handle = svc.acquire_many([_helper_make_request(key, _FlockLockMode.WRITE)])
    fd, _key = handle._fds[0]
    os.close(fd)

    with svc.acquire_many([_helper_make_request(key, _FlockLockMode.WRITE)]):
        pass
