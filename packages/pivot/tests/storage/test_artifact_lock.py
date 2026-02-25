# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false
from __future__ import annotations

import enum
import importlib
import os
import pathlib
import threading
import time
from typing import Any, Protocol, cast

from pivot import loaders, types


class _LockMode(enum.IntEnum):
    READ = 0
    WRITE = 1


class _ArtifactLockModule(Protocol):
    LockMode: type[_LockMode]

    def expand_lock_requests(
        self,
        deps: dict[str, types.ArtifactRef],
        outs: list[types.ArtifactRef],
        project_root: pathlib.Path,
    ) -> list[dict[str, object]]: ...


artifact_lock = cast(
    "_ArtifactLockModule",
    cast("object", importlib.import_module("pivot.storage.artifact_lock")),
)


def _helper_lock_map(requests: list[dict[str, object]]) -> dict[str, object]:
    return {cast("str", request["key"]): request["mode"] for request in requests}


def test_expand_lock_requests_empty() -> None:
    project_root = pathlib.Path("/project")

    result = artifact_lock.expand_lock_requests({}, [], project_root)

    assert result == list[dict[str, object]]()


def test_expand_lock_requests_dep_read_identity_key() -> None:
    project_root = pathlib.Path("/project")
    dep_ref = types.ArtifactRef(
        identity=types.ArtifactIdentity("source_stage", "input"),
        format=loaders.JSON(),
        python_type=dict,
        tag=types.ArtifactTag.DATA,
    )

    result = artifact_lock.expand_lock_requests({"dep": dep_ref}, [], project_root)

    lock_map = _helper_lock_map(result)
    assert lock_map["source_stage:input"] is artifact_lock.LockMode.READ


def test_expand_lock_requests_out_write_identity_key() -> None:
    project_root = pathlib.Path("/project")
    out_ref = types.ArtifactRef(
        identity=types.ArtifactIdentity("train", "metrics"),
        format=loaders.JSON(),
        python_type=dict,
        tag=types.ArtifactTag.METRIC,
    )

    result = artifact_lock.expand_lock_requests({}, [out_ref], project_root)

    lock_map = _helper_lock_map(result)
    assert lock_map["train:metrics"] is artifact_lock.LockMode.WRITE


def test_write_dominates_read() -> None:
    project_root = pathlib.Path("/project")
    ref = types.ArtifactRef(
        identity=types.ArtifactIdentity("stage", "output"),
        format=loaders.JSON(),
        python_type=dict,
        tag=types.ArtifactTag.DATA,
    )

    result = artifact_lock.expand_lock_requests({"output": ref}, [ref], project_root)

    lock_map = _helper_lock_map(result)
    assert lock_map["stage:output"] is artifact_lock.LockMode.WRITE


def test_expand_lock_requests_uses_producer_for_none_key() -> None:
    project_root = pathlib.Path("/project")
    dep_ref = types.ArtifactRef(
        identity=types.ArtifactIdentity("upstream_stage", None),
        format=loaders.JSON(),
        python_type=dict,
        tag=types.ArtifactTag.DATA,
    )

    result = artifact_lock.expand_lock_requests({"dep": dep_ref}, [], project_root)

    lock_map = _helper_lock_map(result)
    assert lock_map["upstream_stage"] is artifact_lock.LockMode.READ


def test_deterministic_sort() -> None:
    project_root = pathlib.Path("/project")
    dep_a = types.ArtifactRef(
        identity=types.ArtifactIdentity("a", "a"),
        format=loaders.JSON(),
        python_type=dict,
        tag=types.ArtifactTag.DATA,
    )
    dep_b = types.ArtifactRef(
        identity=types.ArtifactIdentity("b", "b"),
        format=loaders.JSON(),
        python_type=dict,
        tag=types.ArtifactTag.DATA,
    )

    result = artifact_lock.expand_lock_requests({"b": dep_b, "a": dep_a}, [], project_root)

    keys = [request["key"] for request in result]
    assert keys == ["a:a", "b:b"]


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


def test_flock_acquire_many_normalizes_to_strongest_mode(tmp_path: pathlib.Path) -> None:
    svc = _FlockLockService(tmp_path / "locks")
    requests = [
        _helper_make_request("alpha", _FlockLockMode.READ),
        _helper_make_request("alpha", _FlockLockMode.WRITE),
        _helper_make_request("alpha", _FlockLockMode.READ),
    ]

    handle = svc.acquire_many(requests)
    try:
        assert len(handle._fds) == 1
    finally:
        handle.release()


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
