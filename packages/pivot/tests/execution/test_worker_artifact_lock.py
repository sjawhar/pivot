"""Tests for artifact lock acquisition in worker.execute_stage()."""

from __future__ import annotations

import inspect
import multiprocessing
import os
import pathlib
import queue
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

from pivot import loaders, outputs
from pivot.executor import worker
from pivot.storage import artifact_lock, cache
from pivot.types import OutputMessageKind, StateChange

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _helper_write_output() -> None:
    """Stage that writes an output file."""
    pathlib.Path("out.csv").write_text("a,b\n1,2\n")


def _make_stage_info(
    func: Any,
    tmp_path: pathlib.Path,
    *,
    deps: list[str] | None = None,
    outs: list[outputs.BaseOut] | None = None,
) -> worker.WorkerStageInfo:
    """Create a WorkerStageInfo with sensible defaults for lock tests."""
    expanded_outs = [outputs.require_expanded(out) for out in outs] if outs else []
    return {
        "func": func,
        "fingerprint": {"self:test": "abc123"},
        "deps": deps or [],
        "signature": inspect.signature(func),
        "outs": expanded_outs,
        "params": None,
        "variant": None,
        "overrides": {},
        "checkout_modes": [
            cache.CheckoutMode.HARDLINK,
            cache.CheckoutMode.SYMLINK,
            cache.CheckoutMode.COPY,
        ],
        "run_id": "test_run",
        "force": True,
        "no_commit": True,
        "dep_specs": {},
        "out_specs": {},
        "params_arg_name": None,
        "project_root": tmp_path,
        "state_dir": tmp_path / ".pivot",
    }


def test_lock_acquired_with_correct_keys(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Verify expand_lock_requests is called with deps, outs, project_root."""
    dep_file = tmp_path / "input.csv"
    dep_file.write_text("data")
    (tmp_path / ".pivot").mkdir(parents=True)

    stage_info = _make_stage_info(
        _helper_write_output,
        tmp_path,
        deps=["input.csv"],
        outs=[outputs.Out("out.csv", loader=loaders.PathOnly())],
    )
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)

    # Spy on expand_lock_requests
    spy_expand = mocker.spy(artifact_lock, "expand_lock_requests")

    # Use a mock lock handle that tracks release
    mock_handle = MagicMock(spec=artifact_lock.LockHandle)
    mock_handle.__enter__ = MagicMock(return_value=mock_handle)
    mock_handle.__exit__ = MagicMock(return_value=False)
    mock_acquire = mocker.patch.object(
        artifact_lock.LocalFlockLockService, "acquire_many", return_value=mock_handle
    )

    q: multiprocessing.Queue[Any] = multiprocessing.Queue()
    os.chdir(tmp_path)

    worker.execute_stage("test_stage", stage_info, cache_dir, q)

    # expand_lock_requests called with correct args
    spy_expand.assert_called_once()
    call_args = spy_expand.call_args
    assert call_args[0][0] == ["input.csv"], "deps should match"
    assert call_args[0][2] == tmp_path, "project_root should match"

    # acquire_many called
    mock_acquire.assert_called_once()
    acquired_requests = mock_acquire.call_args[0][0]
    assert len(acquired_requests) > 0, "should have lock requests"


def test_lock_released_on_success(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Verify lock handle __exit__ is called after successful stage execution."""
    (tmp_path / ".pivot").mkdir(parents=True)

    stage_info = _make_stage_info(
        _helper_write_output,
        tmp_path,
        outs=[outputs.Out("out.csv", loader=loaders.PathOnly())],
    )
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)

    mock_handle = MagicMock(spec=artifact_lock.LockHandle)
    mock_handle.__enter__ = MagicMock(return_value=mock_handle)
    mock_handle.__exit__ = MagicMock(return_value=False)
    mocker.patch.object(
        artifact_lock.LocalFlockLockService, "acquire_many", return_value=mock_handle
    )

    q: multiprocessing.Queue[Any] = multiprocessing.Queue()
    os.chdir(tmp_path)

    result = worker.execute_stage("test_stage", stage_info, cache_dir, q)

    assert result["status"].value == "ran", f"expected ran, got {result['status']}"
    mock_handle.__exit__.assert_called_once()


def test_lock_released_on_failure(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Verify lock handle __exit__ is called even when stage fails."""
    (tmp_path / ".pivot").mkdir(parents=True)

    def _helper_failing_stage() -> None:
        raise RuntimeError("boom")

    stage_info = _make_stage_info(
        _helper_failing_stage,
        tmp_path,
        outs=[outputs.Out("out.csv", loader=loaders.PathOnly())],
    )
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)

    mock_handle = MagicMock(spec=artifact_lock.LockHandle)
    mock_handle.__enter__ = MagicMock(return_value=mock_handle)
    mock_handle.__exit__ = MagicMock(return_value=False)
    mocker.patch.object(
        artifact_lock.LocalFlockLockService, "acquire_many", return_value=mock_handle
    )

    q: multiprocessing.Queue[Any] = multiprocessing.Queue()
    os.chdir(tmp_path)

    result = worker.execute_stage("test_stage", stage_info, cache_dir, q)

    assert result["status"].value == "failed", f"expected failed, got {result['status']}"
    assert mock_handle.__exit__.call_count == 1, "lock must be released on failure"


def test_waiting_on_lock_status_message(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Verify WAITING_ON_LOCK status message is sent via on_status callback."""
    (tmp_path / ".pivot").mkdir(parents=True)

    stage_info = _make_stage_info(
        _helper_write_output,
        tmp_path,
        outs=[outputs.Out("out.csv", loader=loaders.PathOnly())],
    )
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)

    # Capture the on_status callback and invoke it during acquire_many
    captured_callback: list[artifact_lock.StatusCallback] = []
    real_handle = artifact_lock.LockHandle(list[tuple[int, str]]())

    def _fake_acquire(
        requests: list[artifact_lock.LockRequest],
        on_status: artifact_lock.StatusCallback | None = None,
    ) -> artifact_lock.LockHandle:
        if on_status is not None:
            captured_callback.append(on_status)
            # Simulate being blocked on a lock
            on_status("data/input.csv", artifact_lock.LockMode.READ, 0.5)
        return real_handle

    mocker.patch.object(
        artifact_lock.LocalFlockLockService, "acquire_many", side_effect=_fake_acquire
    )

    q: multiprocessing.Queue[Any] = multiprocessing.Queue()
    os.chdir(tmp_path)

    worker.execute_stage("test_stage", stage_info, cache_dir, q)

    # Check that on_status was captured and invoked
    assert len(captured_callback) == 1, "on_status callback should have been passed"

    # Drain queue looking for state messages.
    # mp.Queue.empty() is unreliable — use get with timeout instead.
    state_messages: list[StateChange] = []
    while True:
        try:
            msg = q.get(timeout=0.5)
        except queue.Empty:
            break
        if isinstance(msg, dict) and "kind" in msg and msg["kind"] == OutputMessageKind.STATE:
            state_messages.append(cast("StateChange", cast("object", msg)))

    assert len(state_messages) >= 1, "should have emitted waiting_on_lock state message"
    assert state_messages[0]["stage"] == "test_stage"
    assert state_messages[0]["state"] == "waiting_on_lock"
