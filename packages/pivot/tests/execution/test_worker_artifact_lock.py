from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Any

from pivot import executor, outputs
from pivot.storage import artifact_lock

if TYPE_CHECKING:
    import multiprocessing as mp
    import pathlib
    from collections.abc import Callable

    from pytest_mock import MockerFixture

    from pivot.executor import WorkerStageInfo
    from pivot.types import OutputMessage


def _helper_noop_stage() -> None:
    return None


def _helper_failing_stage() -> None:
    raise RuntimeError("boom")


def _make_stage_info(
    func: Callable[..., Any],
    tmp_path: pathlib.Path,
    *,
    deps: list[str] | None = None,
    outs: list[outputs.BaseOut] | None = None,
) -> WorkerStageInfo:
    expanded_outs = [outputs.require_expanded(out) for out in outs] if outs else []
    return {
        "func": func,
        "fingerprint": {"self:test": "abc123"},
        "deps": deps or [],
        "signature": None,
        "outs": expanded_outs,
        "params": None,
        "variant": None,
        "overrides": {},
        "checkout_modes": [],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "dep_specs": {},
        "out_specs": {},
        "params_arg_name": None,
        "project_root": tmp_path,
        "state_dir": tmp_path / ".pivot",
    }


def _drain_output_queue(output_queue: mp.Queue[OutputMessage]) -> list[OutputMessage]:
    items: list[OutputMessage] = []
    while True:
        try:
            items.append(output_queue.get_nowait())
        except queue.Empty:
            break
    return items


def _make_lock_handle(mocker: MockerFixture) -> artifact_lock.LockHandle:
    lock_handle = mocker.MagicMock(spec=artifact_lock.LockHandle)
    lock_handle.__enter__.return_value = lock_handle
    lock_handle.__exit__.return_value = None
    return lock_handle


def test_execute_stage_acquires_artifact_locks(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    stage_info = _make_stage_info(_helper_noop_stage, tmp_path)
    lock_requests = [
        artifact_lock.LockRequest(
            key=str(tmp_path / "artifact.txt"),
            mode=artifact_lock.LockMode.READ,
        )
    ]
    expand_mock = mocker.patch(
        "pivot.executor.worker.artifact_lock.expand_lock_requests",
        autospec=True,
        return_value=lock_requests,
    )
    service_mock = mocker.patch(
        "pivot.executor.worker.artifact_lock.LocalFlockLockService",
        autospec=True,
    )
    lock_handle = _make_lock_handle(mocker)
    service_mock.return_value.acquire_many.return_value = lock_handle

    result = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "ran"
    expand_mock.assert_called_once_with(stage_info["deps"], stage_info["outs"], tmp_path)
    service_mock.assert_called_once_with(stage_info["state_dir"])
    acquire_args, acquire_kwargs = service_mock.return_value.acquire_many.call_args
    assert acquire_args[0] == lock_requests
    assert acquire_kwargs["status_callback"] is not None
    lock_handle.__enter__.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]
    lock_handle.__exit__.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]


def test_execute_stage_emits_waiting_on_lock_state(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    stage_info = _make_stage_info(_helper_noop_stage, tmp_path)
    lock_requests = [
        artifact_lock.LockRequest(
            key=str(tmp_path / "artifact.txt"),
            mode=artifact_lock.LockMode.READ,
        )
    ]
    mocker.patch(
        "pivot.executor.worker.artifact_lock.expand_lock_requests",
        autospec=True,
        return_value=lock_requests,
    )
    service_mock = mocker.patch(
        "pivot.executor.worker.artifact_lock.LocalFlockLockService",
        autospec=True,
    )
    lock_handle = _make_lock_handle(mocker)

    def _acquire_many(
        requests: list[artifact_lock.LockRequest],
        status_callback: Callable[[str, artifact_lock.LockMode, float], None] | None = None,
    ) -> artifact_lock.LockHandle:
        assert status_callback is not None
        status_callback(str(tmp_path / "artifact.txt"), artifact_lock.LockMode.READ, 0.1)
        return lock_handle

    service_mock.return_value.acquire_many.side_effect = _acquire_many

    result = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "ran"
    messages = _drain_output_queue(output_queue)
    state_messages = [msg for msg in messages if msg is not None and msg[0] == "__state__"]
    assert ("__state__", "test_stage", "WAITING_ON_LOCK") in state_messages
    assert ("__state__", "test_stage", "RUNNING") in state_messages


def test_execute_stage_no_waiting_state_when_lock_immediate(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    stage_info = _make_stage_info(_helper_noop_stage, tmp_path)
    lock_requests = [
        artifact_lock.LockRequest(
            key=str(tmp_path / "artifact.txt"),
            mode=artifact_lock.LockMode.READ,
        )
    ]
    mocker.patch(
        "pivot.executor.worker.artifact_lock.expand_lock_requests",
        autospec=True,
        return_value=lock_requests,
    )
    service_mock = mocker.patch(
        "pivot.executor.worker.artifact_lock.LocalFlockLockService",
        autospec=True,
    )
    service_mock.return_value.acquire_many.return_value = _make_lock_handle(mocker)

    result = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "ran"
    messages = _drain_output_queue(output_queue)
    state_messages = [msg for msg in messages if msg is not None and msg[0] == "__state__"]
    assert state_messages == []


def test_execute_stage_releases_lock_on_failure(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    stage_info = _make_stage_info(_helper_failing_stage, tmp_path)
    lock_requests = [
        artifact_lock.LockRequest(
            key=str(tmp_path / "artifact.txt"),
            mode=artifact_lock.LockMode.READ,
        )
    ]
    mocker.patch(
        "pivot.executor.worker.artifact_lock.expand_lock_requests",
        autospec=True,
        return_value=lock_requests,
    )
    service_mock = mocker.patch(
        "pivot.executor.worker.artifact_lock.LocalFlockLockService",
        autospec=True,
    )
    lock_handle = _make_lock_handle(mocker)
    service_mock.return_value.acquire_many.return_value = lock_handle

    result = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "failed"
    lock_handle.__exit__.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]
