from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAny=false
# pyright: reportExplicitAny=false
# pyright: reportUnnecessaryCast=false
import hashlib
import threading
import time
from typing import TYPE_CHECKING, Any, cast

from pivot import loaders as _loaders  # type: ignore[reportMissingImports]
from pivot import outputs as _outputs  # type: ignore[reportMissingImports]
from pivot import path_utils as _path_utils  # type: ignore[reportMissingImports]
from pivot.storage import artifact_lock as _artifact_lock  # type: ignore[reportMissingImports]

if TYPE_CHECKING:
    import pathlib

loaders: Any = _loaders
outputs: Any = _outputs
path_utils: Any = _path_utils
artifact_lock: Any = _artifact_lock


def _make_dir_key(path: pathlib.Path) -> str:
    return path.as_posix() + "/"


def test_expand_empty_inputs_returns_empty_list(tmp_path: pathlib.Path) -> None:
    result = artifact_lock.expand_lock_requests([], [], tmp_path)

    assert result == []


def test_expand_file_dep_adds_read_locks_and_ancestors(tmp_path: pathlib.Path) -> None:
    deps = ["data/input.csv"]

    result = artifact_lock.expand_lock_requests(deps, [], tmp_path)

    file_key = path_utils.canonicalize_artifact_path(deps[0], tmp_path)
    data_dir = tmp_path / "data"
    expected = {
        file_key: artifact_lock.LockMode.READ,
        _make_dir_key(data_dir): artifact_lock.LockMode.READ,
        _make_dir_key(tmp_path): artifact_lock.LockMode.READ,
    }

    result_map = {item["key"]: item["mode"] for item in result}
    assert result_map == expected


def test_expand_file_out_adds_write_lock_and_ancestors(tmp_path: pathlib.Path) -> None:
    out = outputs.Out(path="results/output.csv", loader=loaders.PathOnly())

    result = artifact_lock.expand_lock_requests([], [out], tmp_path)

    file_key = path_utils.canonicalize_artifact_path(out.path, tmp_path)
    results_dir = tmp_path / "results"
    expected = {
        file_key: artifact_lock.LockMode.WRITE,
        _make_dir_key(results_dir): artifact_lock.LockMode.READ,
        _make_dir_key(tmp_path): artifact_lock.LockMode.READ,
    }

    result_map = {item["key"]: item["mode"] for item in result}
    assert result_map == expected


def test_expand_dep_and_out_same_path_write_wins(tmp_path: pathlib.Path) -> None:
    deps = ["shared/data.csv"]
    out = outputs.Out(path="shared/data.csv", loader=loaders.PathOnly())

    result = artifact_lock.expand_lock_requests(deps, [out], tmp_path)

    file_key = path_utils.canonicalize_artifact_path(out.path, tmp_path)
    result_map = {item["key"]: item["mode"] for item in result}
    assert result_map[file_key] == artifact_lock.LockMode.WRITE


def test_expand_directory_out_preserves_trailing_slash(tmp_path: pathlib.Path) -> None:
    directory_loader = cast("Any", loaders.PathOnly())
    out = outputs.DirectoryOut(path="artifacts/", loader=directory_loader)

    result = artifact_lock.expand_lock_requests([], [out], tmp_path)

    dir_key = path_utils.canonicalize_artifact_path(out.path, tmp_path)
    assert dir_key.endswith("/"), "DirectoryOut key must preserve trailing slash"
    result_map = {item["key"]: item["mode"] for item in result}
    assert result_map[dir_key] == artifact_lock.LockMode.WRITE


def test_expand_multiple_keys_sorted_deterministically(tmp_path: pathlib.Path) -> None:
    deps = ["b.txt", "a.txt"]
    outs = [outputs.Out(path="c.txt", loader=loaders.PathOnly())]

    result = artifact_lock.expand_lock_requests(deps, outs, tmp_path)

    keys = [item["key"] for item in result]
    assert keys == sorted(keys), "Lock requests should be sorted by key"


def test_expand_ancestor_dirs_stop_at_project_root(tmp_path: pathlib.Path) -> None:
    deps = ["file.txt"]

    result = artifact_lock.expand_lock_requests(deps, [], tmp_path)

    keys = {item["key"] for item in result}
    assert "/" not in keys, "Should not lock filesystem root"
    assert _make_dir_key(tmp_path) in keys


def _helper_make_request(path: pathlib.Path, mode: Any) -> dict[str, Any]:
    return artifact_lock.LockRequest(key=path.as_posix(), mode=mode)


def test_local_lock_single_process_acquire_release(tmp_path: pathlib.Path) -> None:
    state_dir = tmp_path / "state"
    service = artifact_lock.LocalFlockLockService(state_dir, retry_interval=0.1)
    request = _helper_make_request(tmp_path / "data.txt", artifact_lock.LockMode.WRITE)

    with service.acquire_many([request]):
        pass

    with service.acquire_many([request]):
        pass


def test_local_lock_creates_lock_file_under_state_dir(tmp_path: pathlib.Path) -> None:
    state_dir = tmp_path / "state"
    service = artifact_lock.LocalFlockLockService(state_dir, retry_interval=0.1)
    key_path = tmp_path / "artifact.txt"
    request = _helper_make_request(key_path, artifact_lock.LockMode.READ)

    with service.acquire_many([request]):
        lock_dir = state_dir / "locks"
        expected_name = hashlib.sha256(key_path.as_posix().encode()).hexdigest()
        expected_path = lock_dir / expected_name
        assert expected_path.exists(), "Lock file should exist under state_dir/locks"


def test_local_lock_exclusive_serializes_writes(tmp_path: pathlib.Path) -> None:
    state_dir = tmp_path / "state"
    service = artifact_lock.LocalFlockLockService(state_dir, retry_interval=0.1)
    request = _helper_make_request(tmp_path / "exclusive.txt", artifact_lock.LockMode.WRITE)

    acquired_first = threading.Event()
    release_first = threading.Event()
    acquired_second = threading.Event()
    elapsed_holder: list[float] = []

    def _thread_first() -> None:
        with service.acquire_many([request]):
            acquired_first.set()
            _ = release_first.wait()

    def _thread_second() -> None:
        _ = acquired_first.wait()
        start = time.monotonic()
        with service.acquire_many([request]):
            elapsed_holder.append(time.monotonic() - start)
            acquired_second.set()

    thread_one = threading.Thread(target=_thread_first)
    thread_two = threading.Thread(target=_thread_second)
    thread_one.start()
    _ = acquired_first.wait()
    thread_two.start()

    assert not acquired_second.wait(0.1), "Second writer should block until release"
    release_first.set()

    thread_one.join()
    thread_two.join()
    assert acquired_second.is_set(), "Second writer should eventually acquire lock"
    assert elapsed_holder and elapsed_holder[0] >= 0.1, "Second writer should wait"


def test_local_lock_shared_allows_concurrent_reads(tmp_path: pathlib.Path) -> None:
    state_dir = tmp_path / "state"
    service = artifact_lock.LocalFlockLockService(state_dir, retry_interval=0.1)
    request = _helper_make_request(tmp_path / "shared.txt", artifact_lock.LockMode.READ)

    acquired_first = threading.Event()
    acquired_second = threading.Event()
    release_both = threading.Event()

    def _thread_reader(marker: threading.Event) -> None:
        with service.acquire_many([request]):
            marker.set()
            _ = release_both.wait()

    thread_one = threading.Thread(target=_thread_reader, args=(acquired_first,))
    thread_two = threading.Thread(target=_thread_reader, args=(acquired_second,))
    thread_one.start()
    _ = acquired_first.wait()
    thread_two.start()

    assert acquired_second.wait(0.2), "Second reader should acquire shared lock"
    release_both.set()

    thread_one.join()
    thread_two.join()


def test_local_lock_write_blocks_readers(tmp_path: pathlib.Path) -> None:
    state_dir = tmp_path / "state"
    service = artifact_lock.LocalFlockLockService(state_dir, retry_interval=0.1)
    key_path = tmp_path / "write-blocks-read.txt"
    write_request = _helper_make_request(key_path, artifact_lock.LockMode.WRITE)
    read_request = _helper_make_request(key_path, artifact_lock.LockMode.READ)

    acquired_writer = threading.Event()
    release_writer = threading.Event()
    acquired_reader = threading.Event()

    def _thread_writer() -> None:
        with service.acquire_many([write_request]):
            acquired_writer.set()
            _ = release_writer.wait()

    def _thread_reader() -> None:
        _ = acquired_writer.wait()
        with service.acquire_many([read_request]):
            acquired_reader.set()

    writer_thread = threading.Thread(target=_thread_writer)
    reader_thread = threading.Thread(target=_thread_reader)
    writer_thread.start()
    _ = acquired_writer.wait()
    reader_thread.start()

    assert not acquired_reader.wait(0.1), "Reader should block behind writer"
    release_writer.set()

    writer_thread.join()
    reader_thread.join()
    assert acquired_reader.is_set(), "Reader should acquire after writer releases"


def test_local_lock_status_callback_invoked_on_contention(tmp_path: pathlib.Path) -> None:
    state_dir = tmp_path / "state"
    service = artifact_lock.LocalFlockLockService(state_dir, retry_interval=0.1)
    request = _helper_make_request(tmp_path / "callback.txt", artifact_lock.LockMode.WRITE)

    acquired_first = threading.Event()
    release_first = threading.Event()
    callbacks: list[tuple[str, Any, float]] = []

    def _thread_holder() -> None:
        with service.acquire_many([request]):
            acquired_first.set()
            _ = release_first.wait()

    def _thread_waiter() -> None:
        _ = acquired_first.wait()

        def _status(key: str, mode: Any, elapsed: float) -> None:
            callbacks.append((key, mode, elapsed))

        with service.acquire_many([request], status_callback=_status):
            pass

    holder = threading.Thread(target=_thread_holder)
    waiter = threading.Thread(target=_thread_waiter)
    holder.start()
    _ = acquired_first.wait()
    waiter.start()

    time.sleep(0.25)
    release_first.set()

    holder.join()
    waiter.join()

    assert callbacks, "Status callback should be invoked during contention"
    key, mode, elapsed = callbacks[0]
    assert key == request["key"]
    assert mode == artifact_lock.LockMode.WRITE
    assert elapsed > 0


def test_local_lock_releases_when_fd_closed(tmp_path: pathlib.Path) -> None:
    state_dir = tmp_path / "state"
    service = artifact_lock.LocalFlockLockService(state_dir, retry_interval=0.1)
    request = _helper_make_request(tmp_path / "crash.txt", artifact_lock.LockMode.WRITE)

    handle = service.acquire_many([request])
    held_file = handle._held[0]["file"]
    held_file.close()

    with service.acquire_many([request]):
        pass
