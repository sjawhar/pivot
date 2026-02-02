from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import pathlib
import shutil
import sys
import unicodedata
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

import pytest

from pivot import exceptions, executor, loaders, outputs, registry, stage_def
from pivot.executor import core as executor_core
from pivot.executor import worker
from pivot.storage import cache, lock, state

if TYPE_CHECKING:
    import multiprocessing as mp
    from collections.abc import Callable, Generator

    from pytest_mock import MockerFixture

    from pivot.executor import WorkerStageInfo
    from pivot.types import DirHash, DirManifestEntry, FileHash, OutputHash, OutputMessage


class _PlainParams(stage_def.StageParams):
    """StageParams for testing parameter injection."""

    threshold: float = 0.5


def _make_stage_info(
    func: Callable[..., Any],
    tmp_path: pathlib.Path,
    *,
    fingerprint: dict[str, str] | None = None,
    deps: list[str] | None = None,
    outs: list[outputs.BaseOut] | None = None,
    params: stage_def.StageParams | None = None,
    signature: inspect.Signature | None = None,
    checkout_modes: list[cache.CheckoutMode] | None = None,
    run_id: str = "test_run",
    force: bool = False,
    no_commit: bool = False,
    no_cache: bool = False,
    dep_specs: dict[str, stage_def.FuncDepSpec] | None = None,
    out_specs: dict[str, outputs.BaseOut] | None = None,
    params_arg_name: str | None = None,
) -> WorkerStageInfo:
    """Create a WorkerStageInfo with sensible defaults for testing."""
    return {
        "func": func,
        "fingerprint": fingerprint or {"self:test": "abc123"},
        "deps": deps or [],
        "signature": signature,
        "outs": outs or [],
        "params": params,
        "variant": None,
        "overrides": {},
        "checkout_modes": checkout_modes
        or [cache.CheckoutMode.HARDLINK, cache.CheckoutMode.SYMLINK, cache.CheckoutMode.COPY],
        "run_id": run_id,
        "force": force,
        "no_commit": no_commit,
        "no_cache": no_cache,
        "dep_specs": dep_specs or {},
        "out_specs": out_specs or {},
        "params_arg_name": params_arg_name,
        "project_root": tmp_path,
        "state_dir": tmp_path / ".pivot",
    }


# Helper functions for joblib tests (lambdas can't be typed properly without stubs)
def _helper_double(x: int) -> int:
    """Double the input value."""
    return x * 2


def _helper_identity(x: int) -> int:
    """Return the input value unchanged."""
    return x


def _helper_always_fail_takeover(sentinel: pathlib.Path, stale_pid: int | None) -> bool:
    """Helper that always fails lock takeover (for testing retry exhaustion)."""
    _ = sentinel, stale_pid  # Unused
    return False


# =============================================================================
# execute_stage Tests
# =============================================================================


def test_execute_stage_with_missing_deps(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Worker returns failed status when dependency files are missing."""
    stage_info = _make_stage_info(lambda: None, tmp_path, deps=["missing_file.txt"])
    result = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "failed"
    assert "missing deps" in result["reason"]
    assert "missing_file.txt" in result["reason"]


def test_execute_stage_with_directory_dep(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Worker hashes directory dependency and runs stage."""
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    (data_dir / "file.txt").write_text("content")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("done")

    stage_info = _make_stage_info(stage_func, tmp_path, deps=["data_dir"])
    result = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "ran", f"Expected ran, got {result}"
    assert (tmp_path / "output.txt").read_text() == "done"


def test_execute_stage_runs_unchanged_stage(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Worker skips stage when fingerprint matches and deps unchanged."""
    (tmp_path / "input.txt").write_text("data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("result")

    stage_info = _make_stage_info(
        stage_func, tmp_path, fingerprint={"self:stage_func": "fp123"}, deps=["input.txt"]
    )

    # First run - creates lock file
    result1 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"
    assert (tmp_path / "output.txt").read_text() == "result"

    # Second run - should skip (unchanged)
    result2 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result2["status"] == "skipped"
    assert result2["reason"] == "unchanged"


def test_execute_stage_reruns_when_fingerprint_changes(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Worker reruns stage when code fingerprint changes."""
    (tmp_path / "input.txt").write_text("data")
    counter = tmp_path / "counter.txt"

    def stage_func_v1() -> None:
        count = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(count + 1))

    stage_info_v1 = _make_stage_info(
        stage_func_v1, tmp_path, fingerprint={"self:stage_func_v1": "fp_v1"}, deps=["input.txt"]
    )

    # First run
    result1 = executor.execute_stage("test_stage", stage_info_v1, worker_env, output_queue)
    assert result1["status"] == "ran"
    assert counter.read_text() == "1"

    # Second run with different fingerprint
    stage_info_v2 = _make_stage_info(
        stage_func_v1, tmp_path, fingerprint={"self:stage_func_v1": "fp_v2"}, deps=["input.txt"]
    )
    result2 = executor.execute_stage("test_stage", stage_info_v2, worker_env, output_queue)
    assert result2["status"] == "ran"
    assert result2["reason"] == "Code changed"
    assert counter.read_text() == "2"


def test_execute_stage_handles_stage_exception(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Worker returns failed status when stage raises exception."""
    (tmp_path / "input.txt").write_text("data")

    def failing_stage() -> None:
        raise RuntimeError("Stage failed intentionally")

    stage_info = _make_stage_info(
        failing_stage, tmp_path, fingerprint={"self:failing_stage": "fp123"}, deps=["input.txt"]
    )
    result = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "failed"
    assert "Stage failed intentionally" in result["reason"]


def test_execute_stage_handles_sys_exit(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Worker catches sys.exit and returns failed status."""
    (tmp_path / "input.txt").write_text("data")

    def exits_stage() -> None:
        sys.exit(42)

    stage_info = _make_stage_info(
        exits_stage, tmp_path, fingerprint={"self:exits_stage": "fp123"}, deps=["input.txt"]
    )
    result = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "failed"
    assert "sys.exit" in result["reason"]
    assert "42" in result["reason"]


def test_execute_stage_handles_keyboard_interrupt(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Worker returns failed status for KeyboardInterrupt."""
    (tmp_path / "input.txt").write_text("data")

    def interrupted_stage() -> None:
        raise KeyboardInterrupt("User cancelled")

    stage_info = _make_stage_info(
        interrupted_stage,
        tmp_path,
        fingerprint={"self:interrupted_stage": "fp123"},
        deps=["input.txt"],
    )
    result = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "failed"
    assert "KeyboardInterrupt" in result["reason"]


# =============================================================================
# _run_stage_function_with_injection Tests
# =============================================================================


def test_run_stage_function_captures_stdout(output_queue: mp.Queue[OutputMessage]) -> None:
    """Captures stdout from stage function."""

    def stage_with_output() -> None:
        print("line1")
        print("line2")

    output_lines: list[tuple[str, bool]] = []
    worker._run_stage_function_with_injection(
        stage_with_output, "test_stage", output_queue, output_lines
    )

    assert len(output_lines) == 2
    assert output_lines[0] == ("line1", False)  # stdout
    assert output_lines[1] == ("line2", False)


def test_run_stage_function_captures_stderr(output_queue: mp.Queue[OutputMessage]) -> None:
    """Captures stderr from stage function."""

    def stage_with_errors() -> None:
        print("error1", file=sys.stderr)
        print("error2", file=sys.stderr)

    output_lines: list[tuple[str, bool]] = []
    worker._run_stage_function_with_injection(
        stage_with_errors, "test_stage", output_queue, output_lines
    )

    assert len(output_lines) == 2
    assert output_lines[0] == ("error1", True)  # stderr
    assert output_lines[1] == ("error2", True)


def test_run_stage_function_captures_mixed_output(output_queue: mp.Queue[OutputMessage]) -> None:
    """Captures both stdout and stderr."""

    def stage_mixed() -> None:
        print("stdout1")
        print("stderr1", file=sys.stderr)
        print("stdout2")

    output_lines: list[tuple[str, bool]] = []
    worker._run_stage_function_with_injection(stage_mixed, "test_stage", output_queue, output_lines)

    assert len(output_lines) == 3
    assert output_lines[0] == ("stdout1", False)
    assert output_lines[1] == ("stderr1", True)
    assert output_lines[2] == ("stdout2", False)


def test_run_stage_function_restores_streams(output_queue: mp.Queue[OutputMessage]) -> None:
    """Restores original stdout/stderr after execution."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    def noop_stage() -> None:
        pass

    output_lines: list[tuple[str, bool]] = []
    worker._run_stage_function_with_injection(noop_stage, "test", output_queue, output_lines)

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


def test_run_stage_function_restores_streams_on_exception(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """Restores streams even when stage raises exception."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    def failing_stage() -> None:
        raise RuntimeError("fail")

    output_lines: list[tuple[str, bool]] = []
    with pytest.raises(RuntimeError):
        worker._run_stage_function_with_injection(failing_stage, "test", output_queue, output_lines)

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


def test_run_stage_function_captures_partial_lines(output_queue: mp.Queue[OutputMessage]) -> None:
    """Captures output without trailing newline."""

    def stage_no_newline() -> None:
        sys.stdout.write("no newline")
        sys.stdout.flush()

    output_lines: list[tuple[str, bool]] = []
    worker._run_stage_function_with_injection(
        stage_no_newline, "test_stage", output_queue, output_lines
    )

    assert len(output_lines) == 1
    assert output_lines[0] == ("no newline", False)


# =============================================================================
# _QueueWriter Tests
# =============================================================================


def test_queue_writer_splits_on_newlines(output_queue: mp.Queue[OutputMessage]) -> None:
    """_QueueWriter splits output on newlines."""
    output_lines: list[tuple[str, bool]] = []
    writer = worker._QueueWriter(
        "test_stage", output_queue, is_stderr=False, output_lines=output_lines
    )

    bytes_written = writer.write("line1\nline2\n")

    assert bytes_written == len("line1\nline2\n")
    assert output_lines == [("line1", False), ("line2", False)]


def test_queue_writer_buffers_partial_lines(output_queue: mp.Queue[OutputMessage]) -> None:
    """_QueueWriter buffers incomplete lines."""
    output_lines: list[tuple[str, bool]] = []
    writer = worker._QueueWriter(
        "test_stage", output_queue, is_stderr=False, output_lines=output_lines
    )

    writer.write("partial")
    assert output_lines == []  # Not flushed yet

    writer.write(" line\n")
    assert output_lines == [("partial line", False)]


def test_queue_writer_flush_writes_buffer(output_queue: mp.Queue[OutputMessage]) -> None:
    """_QueueWriter.flush() writes buffered content."""
    output_lines: list[tuple[str, bool]] = []
    writer = worker._QueueWriter(
        "test_stage", output_queue, is_stderr=False, output_lines=output_lines
    )

    writer.write("no newline")
    assert output_lines == []

    writer.flush()
    assert output_lines == [("no newline", False)]


def test_queue_writer_distinguishes_stderr(output_queue: mp.Queue[OutputMessage]) -> None:
    """_QueueWriter marks stderr lines correctly."""
    output_lines: list[tuple[str, bool]] = []
    writer = worker._QueueWriter(
        "test_stage", output_queue, is_stderr=True, output_lines=output_lines
    )

    writer.write("error\n")
    assert output_lines == [("error", True)]


def test_queue_writer_handles_multiple_newlines(output_queue: mp.Queue[OutputMessage]) -> None:
    """_QueueWriter handles text with multiple consecutive newlines."""
    output_lines: list[tuple[str, bool]] = []
    writer = worker._QueueWriter(
        "test_stage", output_queue, is_stderr=False, output_lines=output_lines
    )

    writer.write("line1\n\nline2\n")
    # Empty lines are skipped (code checks 'if line:')
    assert output_lines == [("line1", False), ("line2", False)]


def test_queue_writer_empty_flush_does_nothing(output_queue: mp.Queue[OutputMessage]) -> None:
    """_QueueWriter.flush() with empty buffer does nothing."""
    output_lines: list[tuple[str, bool]] = []
    writer = worker._QueueWriter(
        "test_stage", output_queue, is_stderr=False, output_lines=output_lines
    )

    writer.flush()
    assert output_lines == []


def test_queue_writer_isatty_returns_false(output_queue: mp.Queue[OutputMessage]) -> None:
    """_QueueWriter.isatty() returns False."""
    output_lines: list[tuple[str, bool]] = []
    writer = worker._QueueWriter(
        "test_stage", output_queue, is_stderr=False, output_lines=output_lines
    )

    assert writer.isatty() is False


def test_queue_writer_fileno_raises_unsupported_operation(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """_QueueWriter.fileno() raises io.UnsupportedOperation."""
    output_lines: list[tuple[str, bool]] = []
    writer = worker._QueueWriter(
        "test_stage", output_queue, is_stderr=False, output_lines=output_lines
    )

    with pytest.raises(io.UnsupportedOperation, match="file descriptor"):
        writer.fileno()


def test_queue_writer_context_manager_flushes_on_exit(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """_QueueWriter context manager flushes buffer on exit."""
    output_lines: list[tuple[str, bool]] = []

    with worker._QueueWriter(
        "test_stage", output_queue, is_stderr=False, output_lines=output_lines
    ) as writer:
        writer.write("no newline")
        assert output_lines == []  # Not flushed yet

    # Flushed on context exit
    assert output_lines == [("no newline", False)]


def test_queue_writer_context_manager_flushes_on_exception(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """_QueueWriter context manager flushes buffer even when exception raised."""
    output_lines: list[tuple[str, bool]] = []

    with (
        pytest.raises(RuntimeError),
        worker._QueueWriter("test_stage", output_queue, is_stderr=False, output_lines=output_lines),
    ):
        print("before error")
        raise RuntimeError("test error")

    # Output captured despite exception
    assert output_lines == [("before error", False)]


def test_run_stage_function_preserves_output_on_exception(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """Output is preserved even when stage function raises exception."""

    def failing_stage() -> None:
        print("line before error")
        raise RuntimeError("stage failed")

    output_lines: list[tuple[str, bool]] = []
    with pytest.raises(RuntimeError):
        worker._run_stage_function_with_injection(
            failing_stage, "test_stage", output_queue, output_lines
        )

    # Output captured despite exception
    assert len(output_lines) == 1
    assert output_lines[0] == ("line before error", False)


# =============================================================================
# Execution Lock Tests
# =============================================================================


def test_execution_lock_creates_sentinel_file(worker_env: pathlib.Path) -> None:
    """Execution lock creates sentinel file during execution."""
    sentinel_path = worker_env / "test_stage.running"

    with lock.execution_lock("test_stage", worker_env) as sentinel:
        assert sentinel.exists()
        assert sentinel == sentinel_path
        content = sentinel.read_text()
        assert content.strip().isdigit()  # Just the PID number

    # Cleaned up after context
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "stage_name",
    [
        pytest.param("simple", id="simple"),
        pytest.param("train@model=gpt4", id="matrix-with-equals"),
        pytest.param("process@v1.2", id="matrix-with-dot"),
        pytest.param("stage_with_underscores", id="underscores"),
        pytest.param("stage-with-dashes", id="dashes"),
    ],
)
def test_execution_lock_with_various_stage_names(worker_env: pathlib.Path, stage_name: str) -> None:
    """Execution lock works with various stage name formats including matrix names."""
    sentinel_path = worker_env / f"{stage_name}.running"

    with lock.execution_lock(stage_name, worker_env) as sentinel:
        assert sentinel.exists()
        assert sentinel == sentinel_path

    assert not sentinel.exists()


def test_execution_lock_removes_sentinel_on_exception(worker_env: pathlib.Path) -> None:
    """Execution lock removes sentinel even when exception occurs."""
    sentinel_path = worker_env / "test_stage.running"

    with pytest.raises(RuntimeError), lock.execution_lock("test_stage", worker_env):
        assert sentinel_path.exists()
        raise RuntimeError("intentional")

    assert not sentinel_path.exists()


def test_acquire_execution_lock_succeeds_when_available(worker_env: pathlib.Path) -> None:
    """Acquire lock succeeds when no lock exists."""
    sentinel = lock.acquire_execution_lock("test_stage", worker_env)

    assert sentinel.exists()
    assert sentinel == worker_env / "test_stage.running"

    # Cleanup
    sentinel.unlink()


def test_acquire_execution_lock_fails_when_held_by_live_process(
    worker_env: pathlib.Path,
) -> None:
    """Acquire lock fails when held by a running process."""
    sentinel = worker_env / "test_stage.running"
    sentinel.write_text(str(os.getpid()))

    with pytest.raises(exceptions.StageAlreadyRunningError) as exc_info:
        lock.acquire_execution_lock("test_stage", worker_env)

    assert "already running" in str(exc_info.value)
    assert str(os.getpid()) in str(exc_info.value)

    # Cleanup
    sentinel.unlink()


def test_acquire_execution_lock_breaks_stale_lock(worker_env: pathlib.Path) -> None:
    """Acquire lock breaks stale lock from dead process."""
    sentinel = worker_env / "test_stage.running"
    sentinel.write_text("999999999")  # Non-existent PID

    result_sentinel = lock.acquire_execution_lock("test_stage", worker_env)

    assert result_sentinel.exists()
    assert result_sentinel == sentinel

    # Cleanup
    result_sentinel.unlink()


def test_acquire_execution_lock_breaks_corrupted_lock(worker_env: pathlib.Path) -> None:
    """Acquire lock breaks corrupted lock file."""
    sentinel = worker_env / "test_stage.running"
    sentinel.write_text("corrupted content")

    result_sentinel = lock.acquire_execution_lock("test_stage", worker_env)

    assert result_sentinel.exists()

    # Cleanup
    result_sentinel.unlink()


def test_acquire_execution_lock_breaks_negative_pid_lock(worker_env: pathlib.Path) -> None:
    """Acquire lock breaks lock with invalid negative PID."""
    sentinel = worker_env / "test_stage.running"
    sentinel.write_text("-1")

    result_sentinel = lock.acquire_execution_lock("test_stage", worker_env)

    assert result_sentinel.exists()

    # Cleanup
    result_sentinel.unlink()


# =============================================================================
# Process Alive Check Tests
# =============================================================================


def test_is_process_alive_returns_true_for_self() -> None:
    """is_process_alive returns True for own PID."""
    assert lock._is_process_alive(os.getpid())


def test_is_process_alive_returns_false_for_nonexistent() -> None:
    """is_process_alive returns False for non-existent PID."""
    assert not lock._is_process_alive(999999999)


def test_is_process_alive_returns_true_for_init() -> None:
    """is_process_alive returns True for PID 1 (init/systemd)."""
    # PID 1 always exists (init/systemd)
    assert lock._is_process_alive(1)


# =============================================================================
# _read_lock_pid Tests
# =============================================================================


def test_read_lock_pid_returns_pid_for_valid_file(worker_env: pathlib.Path) -> None:
    """_read_lock_pid extracts PID from valid lock file."""
    sentinel = worker_env / "test.running"
    sentinel.write_text("12345")

    assert lock._read_lock_pid(sentinel) == 12345


def test_read_lock_pid_returns_none_for_missing_file(worker_env: pathlib.Path) -> None:
    """_read_lock_pid returns None for non-existent file."""
    sentinel = worker_env / "nonexistent.running"

    assert lock._read_lock_pid(sentinel) is None


def test_read_lock_pid_returns_none_for_corrupted_file(worker_env: pathlib.Path) -> None:
    """_read_lock_pid returns None for corrupted content."""
    sentinel = worker_env / "test.running"
    sentinel.write_text("garbage content")

    assert lock._read_lock_pid(sentinel) is None


def test_read_lock_pid_returns_none_for_negative_pid(worker_env: pathlib.Path) -> None:
    """_read_lock_pid returns None for invalid negative PID."""
    sentinel = worker_env / "test.running"
    sentinel.write_text("-1")

    assert lock._read_lock_pid(sentinel) is None


def test_read_lock_pid_returns_none_for_zero_pid(worker_env: pathlib.Path) -> None:
    """_read_lock_pid returns None for invalid zero PID."""
    sentinel = worker_env / "test.running"
    sentinel.write_text("0")

    assert lock._read_lock_pid(sentinel) is None


# =============================================================================
# _atomic_lock_takeover Tests
# =============================================================================


def test_atomic_lock_takeover_succeeds_on_stale_lock(worker_env: pathlib.Path) -> None:
    """Atomic takeover creates lock with current process PID."""
    sentinel = worker_env / "test_stage.running"
    sentinel.write_text("999999999")  # Stale lock

    result = lock._atomic_lock_takeover(sentinel, 999999999)

    assert result is True
    assert sentinel.exists()
    content = sentinel.read_text()
    assert content.strip() == str(os.getpid())

    # Cleanup
    sentinel.unlink()


def test_atomic_lock_takeover_fails_when_another_process_wins(
    worker_env: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Atomic takeover returns False when another process beats us."""
    sentinel = worker_env / "test_stage.running"
    original_replace = os.replace

    def sneaky_replace(src: str, dst: str) -> None:
        """Simulate another process winning the race after our rename."""
        original_replace(src, dst)
        # Immediately overwrite with different PID to simulate race
        pathlib.Path(dst).write_text("999888777")

    monkeypatch.setattr(os, "replace", sneaky_replace)
    sentinel.write_text("999999999")  # Stale lock

    result = lock._atomic_lock_takeover(sentinel, 999999999)

    assert result is False

    # Cleanup
    sentinel.unlink()


def test_atomic_takeover_cleans_temp_on_error(
    worker_env: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Temp file is cleaned up when rename fails."""
    sentinel = worker_env / "test_stage.running"
    sentinel.write_text("999999999")

    def failing_replace(src: str, dst: str) -> None:
        raise OSError("Simulated disk error")

    monkeypatch.setattr(os, "replace", failing_replace)

    result = lock._atomic_lock_takeover(sentinel, 999999999)

    assert result is False
    # Verify no temp files left behind
    temp_files = list(worker_env.glob(".test_stage.running.*"))
    assert len(temp_files) == 0, f"Temp files should be cleaned up: {temp_files}"

    # Cleanup
    sentinel.unlink()


def test_acquire_lock_retries_after_failed_takeover(
    worker_env: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock acquisition retries when atomic takeover fails."""
    sentinel = worker_env / "test_stage.running"
    call_count = 0
    my_pid = os.getpid()

    def mock_takeover(sent: pathlib.Path, pid: int | None) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            return False  # Fail first attempt
        # Second attempt: actually create the lock
        sent.write_text(str(my_pid))
        return True

    # Start with stale lock
    sentinel.write_text("999999999")
    monkeypatch.setattr(lock, "_atomic_lock_takeover", mock_takeover)

    result = lock.acquire_execution_lock("test_stage", worker_env)

    assert result == sentinel
    assert call_count >= 2, "Should have retried after failed takeover"

    # Cleanup
    sentinel.unlink()


def test_acquire_lock_exhausts_attempts_and_fails(
    worker_env: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock acquisition fails after exhausting all attempts."""
    sentinel = worker_env / "test_stage.running"
    sentinel.write_text("999999999")

    monkeypatch.setattr(lock, "_atomic_lock_takeover", _helper_always_fail_takeover)

    with pytest.raises(exceptions.StageAlreadyRunningError, match="after 3 attempts"):
        lock.acquire_execution_lock("test_stage", worker_env)

    # Cleanup
    sentinel.unlink()


# =============================================================================
# Multiprocess Race Condition Tests
# =============================================================================


def _race_worker_try_takeover(args: tuple[int, str]) -> str:
    """Module-level worker function for stale lock takeover test.

    Takes a tuple of (worker_id, cache_dir_path) for cross-process pickling.
    """
    import time

    worker_id, cache_dir_str = args
    cache_dir = pathlib.Path(cache_dir_str)
    try:
        sentinel = lock.acquire_execution_lock("race_stage", cache_dir)
        time.sleep(0.05)  # Hold lock briefly
        sentinel.unlink(missing_ok=True)
        return f"{worker_id}:success"
    except exceptions.StageAlreadyRunningError:
        return f"{worker_id}:failed"


def _race_worker_try_fresh_acquire(args: tuple[int, str]) -> tuple[int, str]:
    """Module-level worker function for fresh lock acquisition test.

    Takes a tuple of (worker_id, cache_dir_path) for cross-process pickling.
    """
    import time

    worker_id, cache_dir_str = args
    cache_dir = pathlib.Path(cache_dir_str)
    try:
        sentinel = lock.acquire_execution_lock("fresh_lock_test", cache_dir)
        time.sleep(0.1)  # Hold lock to force others to wait/fail
        sentinel.unlink(missing_ok=True)
        return (worker_id, "success")
    except exceptions.StageAlreadyRunningError:
        return (worker_id, "blocked")


def test_concurrent_stale_lock_takeover_race(worker_env: pathlib.Path) -> None:
    """Multiple processes racing to take over a stale lock - only one should win.

    This tests the real race condition scenario where multiple processes detect
    a stale lock and all try to take it over using atomic replace.
    """
    from concurrent import futures

    NUM_PROCESSES = 5
    cache_dir_str = str(worker_env)

    # Create a stale lock (non-existent PID)
    stale_sentinel = worker_env / "race_stage.running"
    stale_sentinel.write_text("999999999")

    try:
        # Pass both worker_id and cache_dir as tuple for each worker
        args = [(i, cache_dir_str) for i in range(NUM_PROCESSES)]

        with futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES) as pool:
            results = list(pool.map(_race_worker_try_takeover, args))

        successes = [r for r in results if ":success" in r]
        failures = [r for r in results if ":failed" in r]

        # At least one should succeed (first one to get the lock)
        # Others should either fail or succeed after the first one releases
        assert len(successes) >= 1, f"Expected at least 1 success, got {successes}"

        # Total should equal NUM_PROCESSES
        assert len(successes) + len(failures) == NUM_PROCESSES
    finally:
        stale_sentinel.unlink(missing_ok=True)


def test_concurrent_fresh_lock_acquisition(worker_env: pathlib.Path) -> None:
    """Multiple processes racing to acquire a fresh lock - only one should succeed at a time."""
    from concurrent import futures

    NUM_PROCESSES = 3
    cache_dir_str = str(worker_env)
    sentinel_path = worker_env / "fresh_lock_test.running"
    sentinel_path.unlink(missing_ok=True)

    try:
        args = [(i, cache_dir_str) for i in range(NUM_PROCESSES)]

        with futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES) as pool:
            results = list(pool.map(_race_worker_try_fresh_acquire, args))

        # At least one should succeed
        successes = [r for r in results if r[1] == "success"]
        blocked = [r for r in results if r[1] == "blocked"]

        assert len(successes) >= 1, "At least one process should acquire the lock"
        # Total should be NUM_PROCESSES
        assert len(successes) + len(blocked) == NUM_PROCESSES
    finally:
        sentinel_path.unlink(missing_ok=True)


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_hash_dependencies_with_existing_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hash_dependencies hashes existing files as FileHash dicts."""
    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")

    monkeypatch.chdir(tmp_path)
    hashes, missing, unreadable = executor.hash_dependencies(["file1.txt", "file2.txt"])

    assert len(hashes) == 2
    # Keys are now normalized paths (absolute)
    file1_key = str(tmp_path / "file1.txt")
    file2_key = str(tmp_path / "file2.txt")
    assert file1_key in hashes
    assert file2_key in hashes
    # File hashes are FileHash dicts with only 'hash' key
    file_hash = hashes[file1_key]
    assert file_hash is not None
    assert "hash" in file_hash
    assert "manifest" not in file_hash, "Files should not have manifest"
    assert len(missing) == 0
    assert len(unreadable) == 0


def test_hash_dependencies_with_missing_files() -> None:
    """hash_dependencies reports missing files."""
    hashes, missing, unreadable = executor.hash_dependencies(["missing1.txt", "missing2.txt"])

    assert len(hashes) == 0
    assert missing == ["missing1.txt", "missing2.txt"]
    assert len(unreadable) == 0


def test_hash_dependencies_with_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hash_dependencies hashes directories with manifest."""
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    (data_dir / "file.txt").write_text("content")

    monkeypatch.chdir(tmp_path)
    hashes, missing, unreadable = executor.hash_dependencies(["data_dir"])

    assert len(hashes) == 1, "Directory should be hashed"
    # Keys are now normalized paths (absolute)
    data_dir_key = str(tmp_path / "data_dir")
    assert data_dir_key in hashes
    dir_hash = hashes[data_dir_key]
    assert "hash" in dir_hash, "Should have hash key"
    assert "manifest" in dir_hash, "Directory should include manifest"
    # Narrow to DirHash via TypeGuard-style assertion
    assert isinstance(dir_hash.get("manifest"), list)
    manifest: list[DirManifestEntry] = dir_hash["manifest"]
    assert len(manifest) == 1, "Manifest should have one file"
    assert manifest[0]["relpath"] == "file.txt"
    assert len(missing) == 0, "No missing dependencies"
    assert len(unreadable) == 0, "No unreadable dependencies"


def test_hash_file_produces_consistent_hash(tmp_path: pathlib.Path) -> None:
    """hash_file produces same hash for same content."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")

    hash1 = cache.hash_file(file_path)
    hash2 = cache.hash_file(file_path)

    assert hash1 == hash2
    assert len(hash1) == 16  # xxhash64 hexdigest


def test_hash_file_different_for_different_content(tmp_path: pathlib.Path) -> None:
    """hash_file produces different hash for different content."""
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("content1")
    file2.write_text("content2")

    hash1 = cache.hash_file(file1)
    hash2 = cache.hash_file(file2)

    assert hash1 != hash2


# =============================================================================
# Generation Tracking Tests
# =============================================================================


def test_generation_skip_on_second_run(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Second run uses generation-based skip detection."""
    (tmp_path / "input.txt").write_text("data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("result")

    out = outputs.Out(str(tmp_path / "output.txt"), loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps=["input.txt"],
        outs=[out],
    )

    # First run - creates output and records generations
    result1 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"
    assert (tmp_path / "output.txt").read_text() == "result"

    # Second run - should skip via generation check
    result2 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result2["status"] == "skipped"
    # Falls back to hash-based skip because input.txt is external (no generation tracking)
    assert "unchanged" in result2["reason"]


def test_generation_mismatch_triggers_rerun(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Stage re-runs when dependency generation changes."""
    # Create input file (external dependency - no generation tracking)
    (tmp_path / "input.txt").write_text("original")

    def step1_func() -> None:
        data = (tmp_path / "input.txt").read_text()
        (tmp_path / "intermediate.txt").write_text(data.upper())

    def step2_func() -> None:
        data = (tmp_path / "intermediate.txt").read_text()
        (tmp_path / "final.txt").write_text(f"Final: {data}")

    step1_out = outputs.Out(str(tmp_path / "intermediate.txt"), loader=loaders.PathOnly())
    step2_out = outputs.Out(str(tmp_path / "final.txt"), loader=loaders.PathOnly())

    step1_info = _make_stage_info(
        step1_func,
        tmp_path,
        fingerprint={"self:step1": "fp1"},
        deps=["input.txt"],
        outs=[step1_out],
    )
    step2_info = _make_stage_info(
        step2_func,
        tmp_path,
        fingerprint={"self:step2": "fp2"},
        deps=["intermediate.txt"],
        outs=[step2_out],
    )

    # First run - both stages execute
    result1_step1 = executor.execute_stage("step1", step1_info, worker_env, output_queue)
    assert result1_step1["status"] == "ran"
    result1_step2 = executor.execute_stage("step2", step2_info, worker_env, output_queue)
    assert result1_step2["status"] == "ran"
    assert (tmp_path / "final.txt").read_text() == "Final: ORIGINAL"

    # Second run - both should skip
    result2_step1 = executor.execute_stage("step1", step1_info, worker_env, output_queue)
    assert result2_step1["status"] == "skipped"
    result2_step2 = executor.execute_stage("step2", step2_info, worker_env, output_queue)
    assert result2_step2["status"] == "skipped"

    # Change input - step1 should re-run
    (tmp_path / "input.txt").write_text("modified")
    result3_step1 = executor.execute_stage("step1", step1_info, worker_env, output_queue)
    assert result3_step1["status"] == "ran"

    # step2 should re-run because intermediate.txt generation changed
    result3_step2 = executor.execute_stage("step2", step2_info, worker_env, output_queue)
    assert result3_step2["status"] == "ran"
    assert (tmp_path / "final.txt").read_text() == "Final: MODIFIED"


def test_external_file_fallback_to_hash_check(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """External files (no generation) trigger fallback to hash-based check."""
    # Create external input file (not a Pivot output, so no generation)
    (tmp_path / "external_data.txt").write_text("external")

    def stage_func() -> None:
        data = (tmp_path / "external_data.txt").read_text()
        (tmp_path / "output.txt").write_text(data.upper())

    out = outputs.Out(str(tmp_path / "output.txt"), loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps=["external_data.txt"],
        outs=[out],
    )

    # First run
    result1 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"
    assert (tmp_path / "output.txt").read_text() == "EXTERNAL"

    # Second run - should skip (external file has no generation, falls back to hash)
    result2 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result2["status"] == "skipped"

    # Modify external file - should detect change via hash fallback
    (tmp_path / "external_data.txt").write_text("changed")
    result3 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result3["status"] == "ran"
    assert (tmp_path / "output.txt").read_text() == "CHANGED"


def test_deps_list_change_triggers_rerun(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Changing deps list (even with same fingerprint) triggers re-run via hash check.

    Generation tracking only checks current deps, so removing a dep from the list
    could cause incorrect skips. This is mitigated because:
    1. In real usage, deps come from pivot.yaml which affects fingerprint
    2. The hash-based fallback compares full dep_hashes dict which catches changes

    This test verifies the hash-based fallback catches deps list changes.
    """
    (tmp_path / "dep_a.txt").write_text("A")
    (tmp_path / "dep_b.txt").write_text("B")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("done")

    out = outputs.Out(str(tmp_path / "output.txt"), loader=loaders.PathOnly())

    # First run with deps=[A, B]
    stage_info_v1 = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage": "fp1"},
        deps=[str(tmp_path / "dep_a.txt"), str(tmp_path / "dep_b.txt")],
        outs=[out],
    )

    result1 = executor.execute_stage("test_stage", stage_info_v1, worker_env, output_queue)
    assert result1["status"] == "ran"

    # Second run with same config - should skip
    result2 = executor.execute_stage("test_stage", stage_info_v1, worker_env, output_queue)
    assert result2["status"] == "skipped"

    # Third run with deps=[A] only (B removed), DIFFERENT fingerprint
    # This simulates real usage where changing pivot.yaml deps changes fingerprint
    stage_info_v2 = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage": "fp2"},  # Different fingerprint
        deps=[str(tmp_path / "dep_a.txt")],  # B removed
        outs=[out],
    )

    result3 = executor.execute_stage("test_stage", stage_info_v2, worker_env, output_queue)
    assert result3["status"] == "ran", "Fingerprint change should trigger re-run"


def test_deps_list_change_same_fingerprint_detected_by_hash(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Even with same fingerprint, deps list change is caught by hash comparison.

    This is a safety test for the edge case where fingerprint somehow stays same
    but deps list changes. The hash-based fallback should catch this.
    """
    (tmp_path / "dep_a.txt").write_text("A")
    (tmp_path / "dep_b.txt").write_text("B")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("done")

    out = outputs.Out(str(tmp_path / "output.txt"), loader=loaders.PathOnly())

    # First run with deps=[A, B]
    stage_info_v1 = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage": "fp_same"},
        deps=[str(tmp_path / "dep_a.txt"), str(tmp_path / "dep_b.txt")],
        outs=[out],
    )

    result1 = executor.execute_stage("test_stage", stage_info_v1, worker_env, output_queue)
    assert result1["status"] == "ran"

    # Second run with deps=[A] only (B removed), SAME fingerprint
    # Generation tracking would miss this, but hash comparison catches it
    stage_info_v2 = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage": "fp_same"},  # Same fingerprint!
        deps=[str(tmp_path / "dep_a.txt")],  # B removed
        outs=[out],
    )

    result2 = executor.execute_stage("test_stage", stage_info_v2, worker_env, output_queue)
    assert result2["status"] == "ran", (
        "Deps list change should trigger re-run even with same fingerprint"
    )


# =============================================================================
# TOCTOU Prevention Tests
# =============================================================================


def test_skip_acquires_execution_lock(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipped stages still acquire execution lock (TOCTOU prevention).

    This ensures output restoration happens inside the lock, preventing race
    conditions between parallel processes.
    """
    (tmp_path / "input.txt").write_text("data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("result")

    out = outputs.Out(str(tmp_path / "output.txt"), loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps=["input.txt"],
        outs=[out],
    )

    # First run - creates lock file and output
    result1 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"

    # Track if execution lock was acquired during second (skip) run
    lock_acquired = False
    original_execution_lock = lock.execution_lock

    @contextlib.contextmanager
    def tracking_execution_lock(
        stage_name: str, cache_dir: pathlib.Path
    ) -> Generator[pathlib.Path]:
        nonlocal lock_acquired
        lock_acquired = True
        with original_execution_lock(stage_name, cache_dir) as sentinel:
            yield sentinel

    monkeypatch.setattr(lock, "execution_lock", tracking_execution_lock)

    # Second run - should skip but still acquire lock
    result2 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result2["status"] == "skipped"
    assert lock_acquired, "Execution lock should be acquired even when skipping (TOCTOU prevention)"


def test_restore_happens_inside_lock(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output restoration occurs while execution lock is held.

    Verifies the fix for TOCTOU race condition where output could be modified
    between skip decision and restoration.
    """
    (tmp_path / "input.txt").write_text("data")
    output_path = tmp_path / "output.txt"

    def stage_func() -> None:
        output_path.write_text("result")

    out = outputs.Out(str(output_path), loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        fingerprint={"self:stage_func": "fp123"},
        deps=["input.txt"],
        outs=[out],
        checkout_modes=[cache.CheckoutMode.COPY],  # Use copy mode for simpler testing
    )

    # First run - creates lock file and caches output
    result1 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"

    # Delete output to force restoration
    output_path.unlink()

    # Track order of operations
    operations: list[str] = []
    original_execution_lock = lock.execution_lock
    original_restore = worker._restore_outputs_from_cache

    @contextlib.contextmanager
    def tracking_lock(stage_name: str, cache_dir: pathlib.Path) -> Generator[pathlib.Path]:
        operations.append("lock_acquire")
        with original_execution_lock(stage_name, cache_dir) as sentinel:
            yield sentinel
        operations.append("lock_release")

    def tracking_restore(*args: object, **kwargs: object) -> bool:
        operations.append("restore")
        return original_restore(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(lock, "execution_lock", tracking_lock)
    monkeypatch.setattr(worker, "_restore_outputs_from_cache", tracking_restore)

    # Second run - should restore output
    result2 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result2["status"] == "skipped"
    assert output_path.exists(), "Output should be restored"

    # Verify restore happened between lock acquire and release
    assert operations == ["lock_acquire", "restore", "lock_release"], (
        f"Restore should happen inside lock. Got: {operations}"
    )


def test_plain_params_no_auto_load_save(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Plain Pydantic params should still work without auto-load/save."""
    output_file = tmp_path / "output.txt"

    def stage_func(params: _PlainParams) -> None:
        output_file.write_text(f"threshold: {params.threshold}")

    out_spec = outputs.Out(str(output_file), loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        stage_func,
        tmp_path,
        outs=[out_spec],
        params=_PlainParams(),
        signature=inspect.signature(stage_func),
        params_arg_name="params",
    )

    result = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "ran"
    assert output_file.exists()
    assert "threshold: 0.5" in output_file.read_text()


# =============================================================================
# Single Annotated Return Type Tests (GitHub Issue #233)
# =============================================================================


def _stage_with_single_annotated_return() -> Annotated[
    dict[str, str], outputs.Out("single_output.json", loaders.JSON[dict[str, str]]())
]:
    """Stage function with single annotated return type (not TypedDict)."""
    return {"status": "success", "message": "output saved"}


def test_single_annotated_return_saves_output(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Single Annotated[T, Out(...)] return type saves output correctly.

    This tests the worker code path at worker.py:463-470 that handles single
    annotated returns (as opposed to TypedDict returns).

    Regression test for GitHub issue #233.
    """
    # Get the output spec from the single annotated return type
    single_out_spec = stage_def.get_single_output_spec_from_return(
        _stage_with_single_annotated_return
    )
    assert single_out_spec is not None, "Should have single output spec"

    stage_info = _make_stage_info(
        _stage_with_single_annotated_return,
        tmp_path,
        fingerprint={"self:_stage_with_single_annotated_return": "fp123"},
        out_specs={"_single": single_out_spec},  # Single return uses "_single" key convention
    )

    result = executor.execute_stage("test_single_return", stage_info, worker_env, output_queue)

    assert result["status"] == "ran", f"Expected ran, got {result}"

    # Verify the output file was created via annotation-based save
    output_file = tmp_path / "single_output.json"
    assert output_file.exists(), "Output file should be created from single annotated return"

    with open(output_file) as f:
        content = json.load(f)
    assert content == {"status": "success", "message": "output saved"}


# =============================================================================
# Nested Parallelism Protection Tests
# =============================================================================


def test_joblib_protection_uses_threading_backend() -> None:
    """Nested joblib.Parallel uses threading backend when executed under protection."""
    from pivot.executor import worker

    def stage_that_checks_backend() -> str:
        from joblib import Parallel, delayed

        # Run a simple parallel job - the backend should be threading
        results = Parallel(n_jobs=2)(delayed(_helper_double)(i) for i in range(4))
        return str(results)

    result = worker._execute_with_joblib_protection(stage_that_checks_backend, {})
    assert result == "[0, 2, 4, 6]", "Parallel execution should work with threading backend"


def test_joblib_protection_noop_without_joblib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protection is a no-op when joblib is not installed."""
    import builtins
    import sys

    from pivot.executor import worker

    # Remove joblib from sys.modules to force re-import
    joblib_modules = [k for k in sys.modules if k == "joblib" or k.startswith("joblib.")]
    for mod in joblib_modules:
        monkeypatch.delitem(sys.modules, mod)

    original_import = builtins.__import__

    def mock_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "joblib":
            raise ImportError("Simulated missing joblib")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    def simple_func() -> str:
        return "executed"

    result = worker._execute_with_joblib_protection(simple_func, {})
    assert result == "executed", "Function should execute normally without joblib"


def test_joblib_protection_uses_threading_backend_via_config(mocker: MockerFixture) -> None:
    """Verifies parallel_config is called with threading backend."""
    from pivot.executor import worker

    def check_func() -> str:
        return "done"

    mock_config = mocker.patch("joblib.parallel_config", autospec=True)
    mock_config.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_config.return_value.__exit__ = mocker.Mock(return_value=None)

    worker._execute_with_joblib_protection(check_func, {})

    mock_config.assert_called_once()
    call_kwargs = mock_config.call_args.kwargs
    assert call_kwargs["backend"] == "threading"


def test_user_explicit_parallel_config_overrides_protection() -> None:
    """User's explicit parallel_config inside their function overrides our defaults.

    NOTE: This test verifies joblib's nested context manager behavior (inner wins).
    If joblib changes this behavior in a future version, this test may need updating.
    """
    from pivot.executor import worker

    def stage_with_explicit_config() -> int:
        from joblib import Parallel, delayed, parallel_config

        # User explicitly requests loky backend inside their code
        with parallel_config(backend="loky", n_jobs=2):
            # This inner config should win (inner context takes precedence)
            results = list(Parallel()(delayed(_helper_identity)(i) for i in range(2)))
        return len(results)

    # Even though we wrap with threading, the user's inner config takes precedence
    result = worker._execute_with_joblib_protection(stage_with_explicit_config, {})
    assert result == 2, "Inner parallel_config should work"


# =============================================================================
# _QueueWriter Thread Safety Tests
# =============================================================================


@pytest.mark.slow
def test_queue_writer_thread_safety_concurrent_writes(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """_QueueWriter handles concurrent writes from multiple threads."""
    import concurrent.futures
    import threading

    output_lines: list[tuple[str, bool]] = []

    writer = worker._QueueWriter(
        "test_stage",
        output_queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    num_threads = 10
    lines_per_thread = 100
    barrier = threading.Barrier(num_threads)

    def write_lines(thread_id: int) -> None:
        # Wait for all threads to be ready before starting
        barrier.wait()
        for i in range(lines_per_thread):
            writer.write(f"thread-{thread_id}-line-{i}\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(write_lines, tid) for tid in range(num_threads)]
        for f in futures:
            f.result()

    writer.flush()

    # All lines should be captured (no data loss from race conditions)
    expected_line_count = num_threads * lines_per_thread
    assert len(output_lines) == expected_line_count, (
        f"Expected {expected_line_count} lines, got {len(output_lines)} - possible thread safety issue"
    )


@pytest.mark.slow
def test_queue_writer_thread_safety_atomic_writes(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """_QueueWriter write() calls are atomic - individual lines are not corrupted.

    When multiple threads write complete lines (ending with newline), each line
    should be intact. Interleaving of partial writes (without newlines) across
    threads is expected since the lock protects individual write() calls, not
    sequences of them.
    """
    import concurrent.futures
    import threading

    output_lines: list[tuple[str, bool]] = []

    writer = worker._QueueWriter(
        "test_stage",
        output_queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    num_threads = 5
    lines_per_thread = 10
    barrier = threading.Barrier(num_threads)

    def write_complete_lines(thread_id: int) -> None:
        barrier.wait()
        # Write complete lines (full line in single write call) - these should NOT interleave
        for i in range(lines_per_thread):
            writer.write(f"t{thread_id}-i{i}\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(write_complete_lines, tid) for tid in range(num_threads)]
        for f in futures:
            f.result()

    writer.flush()

    # All lines should be captured
    expected_count = num_threads * lines_per_thread
    assert len(output_lines) == expected_count, (
        f"Expected {expected_count} lines, got {len(output_lines)}"
    )

    # Each line should be properly formatted (not corrupted by interleaving)
    for line, is_stderr in output_lines:
        assert not is_stderr
        # Line format: t{thread_id}-i{iteration}
        assert line.startswith("t"), f"Corrupted line: {line}"
        parts = line.split("-")
        assert len(parts) == 2, f"Corrupted line (wrong number of parts): {line}"
        assert parts[1].startswith("i"), f"Corrupted line (missing i prefix): {line}"


# =============================================================================
# Joblib Protection Environment Variable Tests
# =============================================================================


def test_joblib_protection_env_var_processes_disables_memmapping(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """PIVOT_NESTED_PARALLELISM=processes uses loky backend with memmapping disabled."""
    from pivot.executor import worker

    def stage_func() -> str:
        return "done"

    monkeypatch.setenv("PIVOT_NESTED_PARALLELISM", "processes")

    mock_config = mocker.patch("joblib.parallel_config", autospec=True)
    mock_config.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_config.return_value.__exit__ = mocker.Mock(return_value=None)

    result = worker._execute_with_joblib_protection(stage_func, {})

    # parallel_config should be called with loky backend and memmapping disabled
    mock_config.assert_called_once_with(backend="loky", max_nbytes=None)
    assert result == "done"


def test_joblib_protection_default_uses_threading(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Default behavior (no env var) uses threading backend."""
    from pivot.executor import worker

    def stage_func() -> str:
        return "done"

    # Ensure env var is not set
    monkeypatch.delenv("PIVOT_NESTED_PARALLELISM", raising=False)

    mock_config = mocker.patch("joblib.parallel_config", autospec=True)
    mock_config.return_value.__enter__ = mocker.Mock(return_value=None)
    mock_config.return_value.__exit__ = mocker.Mock(return_value=None)

    result = worker._execute_with_joblib_protection(stage_func, {})

    mock_config.assert_called_once()
    call_kwargs = mock_config.call_args.kwargs
    assert call_kwargs["backend"] == "threading"
    assert result == "done"


# =============================================================================
# DirectoryOut Tests
# =============================================================================


class _DirectoryOutResult(TypedDict):
    """TypedDict for DirectoryOut test stage."""

    task_results: Annotated[
        dict[str, dict[str, int]], outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    ]


def _directory_out_stage() -> _DirectoryOutResult:
    """Stage that returns DirectoryOut with multiple files."""
    return _DirectoryOutResult(
        task_results={
            "task_a.json": {"accuracy": 95},
            "task_b.json": {"accuracy": 87},
            "subdir/task_c.json": {"accuracy": 92},
        }
    )


def test_directory_out_first_run_writes_files(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """DirectoryOut stage writes all files on first run."""
    out_specs = stage_def.get_output_specs_from_return(_directory_out_stage, "test_stage")
    # Create outs list with absolute path (preserving trailing slash)
    dir_out: outputs.BaseOut = outputs.DirectoryOut(
        str(tmp_path / "results") + "/", loaders.JSON[dict[str, int]]()
    )

    stage_info = _make_stage_info(
        _directory_out_stage,
        tmp_path,
        fingerprint={"self:_directory_out_stage": "fp123"},
        out_specs=out_specs,
        outs=[dir_out],
    )

    result = executor.execute_stage("test_dir_out", stage_info, worker_env, output_queue)

    assert result["status"] == "ran", f"Expected ran, got {result}"

    # Verify all files were created
    results_dir = tmp_path / "results"
    assert results_dir.is_dir(), "Results directory should exist"
    assert (results_dir / "task_a.json").exists()
    assert (results_dir / "task_b.json").exists()
    assert (results_dir / "subdir" / "task_c.json").exists()

    # Verify content
    assert json.loads((results_dir / "task_a.json").read_text()) == {"accuracy": 95}
    assert json.loads((results_dir / "task_b.json").read_text()) == {"accuracy": 87}


def test_directory_out_skipped_on_second_run(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """DirectoryOut stage is skipped on second run when unchanged."""
    out_specs = stage_def.get_output_specs_from_return(_directory_out_stage, "test_stage")
    dir_out: outputs.BaseOut = outputs.DirectoryOut(
        str(tmp_path / "results") + "/", loaders.JSON[dict[str, int]]()
    )

    stage_info = _make_stage_info(
        _directory_out_stage,
        tmp_path,
        fingerprint={"self:_directory_out_stage": "fp123"},
        out_specs=out_specs,
        outs=[dir_out],
    )

    # First run
    result1 = executor.execute_stage("test_dir_out", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"

    # Second run - should skip
    result2 = executor.execute_stage("test_dir_out", stage_info, worker_env, output_queue)
    assert result2["status"] == "skipped", f"Expected skipped, got {result2}"
    assert "unchanged" in result2["reason"]


def test_directory_out_reruns_on_fingerprint_change(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """DirectoryOut stage re-runs when code fingerprint changes."""
    out_specs = stage_def.get_output_specs_from_return(_directory_out_stage, "test_stage")
    dir_out: outputs.BaseOut = outputs.DirectoryOut(
        str(tmp_path / "results") + "/", loaders.JSON[dict[str, int]]()
    )

    stage_info1 = _make_stage_info(
        _directory_out_stage,
        tmp_path,
        fingerprint={"self:_directory_out_stage": "fp123"},
        out_specs=out_specs,
        outs=[dir_out],
    )

    # First run
    result1 = executor.execute_stage("test_dir_out", stage_info1, worker_env, output_queue)
    assert result1["status"] == "ran"

    # Change fingerprint
    stage_info2 = _make_stage_info(
        _directory_out_stage,
        tmp_path,
        fingerprint={"self:_directory_out_stage": "fp456_changed"},
        out_specs=out_specs,
        outs=[dir_out],
    )

    # Second run - should re-run due to fingerprint change
    result2 = executor.execute_stage("test_dir_out", stage_info2, worker_env, output_queue)
    assert result2["status"] == "ran", f"Expected ran, got {result2}"
    assert "Code changed" in result2["reason"]


def test_directory_out_restored_from_cache(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """DirectoryOut files are restored from cache when missing."""
    out_specs = stage_def.get_output_specs_from_return(_directory_out_stage, "test_stage")
    dir_out: outputs.BaseOut = outputs.DirectoryOut(
        str(tmp_path / "results") + "/", loaders.JSON[dict[str, int]]()
    )

    stage_info = _make_stage_info(
        _directory_out_stage,
        tmp_path,
        fingerprint={"self:_directory_out_stage": "fp123"},
        out_specs=out_specs,
        outs=[dir_out],
    )

    # First run - creates files and caches them
    result1 = executor.execute_stage("test_dir_out", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"

    # Verify files exist
    results_dir = tmp_path / "results"
    assert (results_dir / "task_a.json").exists()

    # Delete the output directory
    shutil.rmtree(results_dir)
    assert not results_dir.exists()

    # Second run - should skip and restore from cache
    result2 = executor.execute_stage("test_dir_out", stage_info, worker_env, output_queue)
    assert result2["status"] == "skipped", f"Expected skipped, got {result2}"

    # Verify files were restored
    assert results_dir.exists()
    assert (results_dir / "task_a.json").exists()
    assert (results_dir / "task_b.json").exists()
    assert (results_dir / "subdir" / "task_c.json").exists()

    # Verify content
    assert json.loads((results_dir / "task_a.json").read_text()) == {"accuracy": 95}


# =============================================================================
# Unit Tests for Internal Functions
# =============================================================================


def test_normalize_out_path_preserves_trailing_slash() -> None:
    """_normalize_out_path preserves trailing slash for DirectoryOut paths."""
    result = worker._normalize_out_path("results/")
    assert result.endswith("/"), "Should preserve trailing slash for DirectoryOut"
    assert "results" in result

    result2 = worker._normalize_out_path("a/b/c/")
    assert result2.endswith("/"), "Should preserve trailing slash for nested DirectoryOut"


def test_normalize_out_path_no_slash_for_files() -> None:
    """_normalize_out_path doesn't add trailing slash for regular Out paths."""
    result = worker._normalize_out_path("output.csv")
    assert not result.endswith("/"), "Should not add trailing slash for files"
    assert result.endswith("output.csv")


def test_normalize_out_path_normalizes_relative_paths() -> None:
    """_normalize_out_path normalizes relative path components."""
    # Note: This depends on project.normalize_path behavior
    result = worker._normalize_out_path("./results/")
    assert result.endswith("/"), "Should preserve trailing slash"
    assert "//" not in result, "Should not have double slashes"


def test_directory_needs_restore_returns_false_when_matching(tmp_path: pathlib.Path) -> None:
    """_directory_needs_restore returns False when directory matches manifest."""
    # Create directory with files
    dir_path = tmp_path / "results"
    dir_path.mkdir()
    (dir_path / "a.json").write_text('{"value": 1}')
    (dir_path / "b.json").write_text('{"value": 2}')

    # Use hash_directory to get the actual tree hash and manifest
    tree_hash, manifest = cache.hash_directory(dir_path)
    cached_hash: DirHash = {"hash": tree_hash, "manifest": manifest}

    assert not worker._directory_needs_restore(dir_path, cached_hash)


def test_directory_needs_restore_returns_true_for_missing_file(tmp_path: pathlib.Path) -> None:
    """_directory_needs_restore returns True when a file is missing."""
    dir_path = tmp_path / "results"
    dir_path.mkdir()
    (dir_path / "a.json").write_text('{"value": 1}')
    (dir_path / "b.json").write_text('{"value": 2}')

    # Get hash when both files exist
    tree_hash, manifest = cache.hash_directory(dir_path)
    cached_hash: DirHash = {"hash": tree_hash, "manifest": manifest}

    # Now delete b.json - directory no longer matches
    (dir_path / "b.json").unlink()

    assert worker._directory_needs_restore(dir_path, cached_hash)


def test_directory_needs_restore_returns_true_for_extra_file(tmp_path: pathlib.Path) -> None:
    """_directory_needs_restore returns True when there are extra files."""
    dir_path = tmp_path / "results"
    dir_path.mkdir()
    (dir_path / "a.json").write_text('{"value": 1}')

    # Get hash with only a.json
    tree_hash, manifest = cache.hash_directory(dir_path)
    cached_hash: DirHash = {"hash": tree_hash, "manifest": manifest}

    # Now add extra file - directory no longer matches
    (dir_path / "extra.json").write_text('{"extra": true}')

    assert worker._directory_needs_restore(dir_path, cached_hash)


def test_directory_needs_restore_returns_true_for_wrong_content(tmp_path: pathlib.Path) -> None:
    """_directory_needs_restore returns True when file content doesn't match."""
    dir_path = tmp_path / "results"
    dir_path.mkdir()
    (dir_path / "a.json").write_text('{"value": 1}')

    # Get hash with original content
    tree_hash, manifest = cache.hash_directory(dir_path)
    cached_hash: DirHash = {"hash": tree_hash, "manifest": manifest}

    # Now modify the file - directory no longer matches
    (dir_path / "a.json").write_text('{"value": 999}')

    assert worker._directory_needs_restore(dir_path, cached_hash)


def test_directory_needs_restore_ignores_symlinks(tmp_path: pathlib.Path) -> None:
    """_directory_needs_restore ignores symlinks in directory."""
    dir_path = tmp_path / "results"
    dir_path.mkdir()
    (dir_path / "a.json").write_text('{"value": 1}')

    # Get hash before adding symlink
    tree_hash, manifest = cache.hash_directory(dir_path)
    cached_hash: DirHash = {"hash": tree_hash, "manifest": manifest}

    # Create symlink (should be ignored by hash_directory, so hash should still match)
    symlink_target = tmp_path / "target.json"
    symlink_target.write_text('{"target": true}')
    (dir_path / "link.json").symlink_to(symlink_target)

    # Should not consider symlink as an extra file
    assert not worker._directory_needs_restore(dir_path, cached_hash)


def test_directory_needs_restore_handles_nested_files(tmp_path: pathlib.Path) -> None:
    """_directory_needs_restore correctly handles nested subdirectories."""
    dir_path = tmp_path / "results"
    dir_path.mkdir()
    (dir_path / "top.json").write_text('{"level": "top"}')
    (dir_path / "sub").mkdir()
    (dir_path / "sub" / "nested.json").write_text('{"level": "nested"}')

    # Use hash_directory to get the actual tree hash
    tree_hash, manifest = cache.hash_directory(dir_path)
    cached_hash: DirHash = {"hash": tree_hash, "manifest": manifest}

    assert not worker._directory_needs_restore(dir_path, cached_hash)


def test_directory_needs_restore_empty_manifest(tmp_path: pathlib.Path) -> None:
    """_directory_needs_restore returns True when manifest is empty but files exist."""
    dir_path = tmp_path / "results"
    dir_path.mkdir()

    # Get hash of empty directory
    empty_hash, _ = cache.hash_directory(dir_path)
    cached_hash: DirHash = {"hash": empty_hash, "manifest": []}

    # Now add a file - directory no longer matches
    (dir_path / "a.json").write_text('{"value": 1}')

    # Directory has files but cached hash is for empty directory - needs restore
    assert worker._directory_needs_restore(dir_path, cached_hash)


def test_directory_needs_restore_empty_directory_empty_manifest(tmp_path: pathlib.Path) -> None:
    """_directory_needs_restore returns False when both directory and manifest are empty."""
    dir_path = tmp_path / "results"
    dir_path.mkdir()

    # Get actual hash of empty directory
    tree_hash, manifest = cache.hash_directory(dir_path)
    cached_hash: DirHash = {"hash": tree_hash, "manifest": manifest}

    # Both empty - matches
    assert not worker._directory_needs_restore(dir_path, cached_hash)


# =============================================================================
# _cleanup_restored_paths Tests
# =============================================================================


def test_cleanup_restored_paths_removes_files(tmp_path: pathlib.Path) -> None:
    """_cleanup_restored_paths removes files that were partially restored."""
    # Create some files that represent partially restored outputs
    file1 = tmp_path / "output1.txt"
    file2 = tmp_path / "output2.txt"
    file1.write_text("content1")
    file2.write_text("content2")

    restored_paths = [file1, file2]
    worker._cleanup_restored_paths(restored_paths)

    assert not file1.exists(), "File1 should be removed"
    assert not file2.exists(), "File2 should be removed"


def test_cleanup_restored_paths_removes_directories(tmp_path: pathlib.Path) -> None:
    """_cleanup_restored_paths removes directories and their contents."""
    # Create a directory with files
    dir_path = tmp_path / "results"
    dir_path.mkdir()
    (dir_path / "a.json").write_text('{"value": 1}')
    (dir_path / "b.json").write_text('{"value": 2}')

    worker._cleanup_restored_paths([dir_path])

    assert not dir_path.exists(), "Directory should be removed"


def test_cleanup_restored_paths_handles_nonexistent(tmp_path: pathlib.Path) -> None:
    """_cleanup_restored_paths gracefully handles paths that don't exist."""
    # This should not raise an exception
    nonexistent = tmp_path / "does_not_exist.txt"
    worker._cleanup_restored_paths([nonexistent])


def test_cleanup_restored_paths_handles_mixed(tmp_path: pathlib.Path) -> None:
    """_cleanup_restored_paths handles mix of files, dirs, and nonexistent."""
    file1 = tmp_path / "file.txt"
    file1.write_text("content")
    dir1 = tmp_path / "dir"
    dir1.mkdir()
    (dir1 / "nested.txt").write_text("nested")
    nonexistent = tmp_path / "missing"

    worker._cleanup_restored_paths([file1, dir1, nonexistent])

    assert not file1.exists()
    assert not dir1.exists()


def test_cleanup_restored_paths_empty_list() -> None:
    """_cleanup_restored_paths handles empty list."""
    # Should not raise
    worker._cleanup_restored_paths([])


# =============================================================================
# _directory_needs_restore Error Handling Tests
# =============================================================================


def test_directory_needs_restore_returns_true_on_permission_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_directory_needs_restore returns True when hash computation fails."""
    dir_path = tmp_path / "results"
    dir_path.mkdir()
    (dir_path / "a.json").write_text('{"value": 1}')

    # Get hash before breaking things
    tree_hash, manifest = cache.hash_directory(dir_path)
    cached_hash: DirHash = {"hash": tree_hash, "manifest": manifest}

    # Make hash_directory raise OSError (accepts optional state_db and ignore_filter)
    def failing_hash_directory(
        path: pathlib.Path,
        state_db: state.StateDB | None = None,
        ignore_filter: object = None,
    ) -> tuple[str, list[DirManifestEntry]]:
        raise OSError("Permission denied")

    monkeypatch.setattr(cache, "hash_directory", failing_hash_directory)

    # Should return True (needs restore) when hash fails
    assert worker._directory_needs_restore(dir_path, cached_hash)


def test_directory_needs_restore_handles_unicode_normalization(tmp_path: pathlib.Path) -> None:
    """_directory_needs_restore handles unicode normalization (NFC vs NFD)."""
    dir_path = tmp_path / "results"
    dir_path.mkdir()

    # Create file with combining character (could be stored as NFD on some filesystems)
    # café with combining acute accent (NFD form)
    nfd_name = "cafe\u0301.json"
    # café with precomposed é (NFC form) - unused but documents the equivalence
    _ = unicodedata.normalize("NFC", nfd_name)

    (dir_path / nfd_name).write_text('{"value": 1}')

    # Use hash_directory to get proper hash (handles unicode internally)
    tree_hash, manifest = cache.hash_directory(dir_path)
    cached_hash: DirHash = {"hash": tree_hash, "manifest": manifest}

    # Should handle unicode normalization correctly
    assert not worker._directory_needs_restore(dir_path, cached_hash)


# =============================================================================
# _file_needs_restore Tests
# =============================================================================


def test_file_needs_restore_returns_false_when_matching(tmp_path: pathlib.Path) -> None:
    """_file_needs_restore returns False when file matches cached hash."""
    file_path = tmp_path / "output.txt"
    file_path.write_text("content")

    file_hash = cache.hash_file(file_path)
    cached_hash: FileHash = {"hash": file_hash}

    assert not worker._file_needs_restore(file_path, cached_hash)


def test_file_needs_restore_returns_true_for_missing_file(tmp_path: pathlib.Path) -> None:
    """_file_needs_restore returns True when file doesn't exist."""
    file_path = tmp_path / "missing.txt"
    cached_hash: FileHash = {"hash": "1234567890abcdef"}

    assert worker._file_needs_restore(file_path, cached_hash)


def test_file_needs_restore_returns_true_for_wrong_content(tmp_path: pathlib.Path) -> None:
    """_file_needs_restore returns True when file content doesn't match hash."""
    file_path = tmp_path / "output.txt"
    file_path.write_text("original")

    # Get hash of original content
    original_hash = cache.hash_file(file_path)
    cached_hash: FileHash = {"hash": original_hash}

    # Modify content
    file_path.write_text("modified")

    assert worker._file_needs_restore(file_path, cached_hash)


def test_file_needs_restore_returns_true_on_permission_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_file_needs_restore returns True when hash computation fails."""
    file_path = tmp_path / "output.txt"
    file_path.write_text("content")

    # Get hash before breaking things
    file_hash = cache.hash_file(file_path)
    cached_hash: FileHash = {"hash": file_hash}

    # Make hash_file raise OSError
    def failing_hash(path: pathlib.Path, state_db: state.StateDB | None = None) -> str:
        raise OSError("Permission denied")

    monkeypatch.setattr(cache, "hash_file", failing_hash)

    assert worker._file_needs_restore(file_path, cached_hash)


def test_file_needs_restore_handles_empty_file(tmp_path: pathlib.Path) -> None:
    """_file_needs_restore correctly handles empty files."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")

    empty_hash = cache.hash_file(file_path)
    cached_hash: FileHash = {"hash": empty_hash}

    # Empty file matches
    assert not worker._file_needs_restore(file_path, cached_hash)

    # Now put content - should need restore
    file_path.write_text("not empty anymore")
    assert worker._file_needs_restore(file_path, cached_hash)


def test_file_needs_restore_uses_state_db(tmp_path: pathlib.Path) -> None:
    """_file_needs_restore accepts state_db parameter for hash caching."""
    file_path = tmp_path / "output.txt"
    file_path.write_text("content")
    db_path = tmp_path / "state.db"

    with state.StateDB(db_path) as db:
        file_hash = cache.hash_file(file_path, db)
        cached_hash: FileHash = {"hash": file_hash}

        # Should work with state_db
        assert not worker._file_needs_restore(file_path, cached_hash, state_db=db)


# =============================================================================
# _normalize_out_path Additional Tests
# =============================================================================


def test_normalize_out_path_handles_absolute_path() -> None:
    """_normalize_out_path handles absolute paths."""
    result = worker._normalize_out_path("/absolute/path/output.csv")
    assert not result.endswith("/"), "Files should not have trailing slash"
    assert "output.csv" in result


def test_normalize_out_path_handles_absolute_dir_path() -> None:
    """_normalize_out_path preserves trailing slash for absolute directory paths."""
    result = worker._normalize_out_path("/absolute/path/results/")
    assert result.endswith("/"), "Directory paths should preserve trailing slash"


def test_normalize_out_path_handles_empty_trailing_component() -> None:
    """_normalize_out_path handles paths with multiple trailing slashes."""
    # Single trailing slash
    result = worker._normalize_out_path("results/")
    assert result.endswith("/")
    assert "//" not in result


# =============================================================================
# Run Cache with DirectoryOut Tests
# =============================================================================


class _RunCacheDirectoryOutResult(TypedDict):
    """TypedDict for run cache DirectoryOut test stage."""

    results: Annotated[
        dict[str, dict[str, int]], outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    ]


def _run_cache_directory_stage() -> _RunCacheDirectoryOutResult:
    """Stage that returns DirectoryOut for run cache testing."""
    return _RunCacheDirectoryOutResult(
        results={
            "a.json": {"value": 1},
            "b.json": {"value": 2},
        }
    )


def test_run_cache_skip_with_directory_out(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Run cache correctly handles DirectoryOut restoration."""
    out_specs = stage_def.get_output_specs_from_return(_run_cache_directory_stage, "test_stage")
    dir_out: outputs.BaseOut = outputs.DirectoryOut(
        str(tmp_path / "results") + "/", loaders.JSON[dict[str, int]]()
    )

    stage_info = _make_stage_info(
        _run_cache_directory_stage,
        tmp_path,
        fingerprint={"self:_run_cache_directory_stage": "fp123"},
        out_specs=out_specs,
        outs=[dir_out],
    )

    # First run - creates files and records in run cache
    result1 = executor.execute_stage("test_run_cache_dir", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"

    results_dir = tmp_path / "results"
    assert (results_dir / "a.json").exists()
    assert (results_dir / "b.json").exists()

    # Manually write the run cache entry (normally done by coordinator)
    deferred = result1.get("deferred_writes", {})
    if "run_cache_input_hash" in deferred and "run_cache_entry" in deferred:
        state_db = state.StateDB(tmp_path / ".pivot" / "state.db")
        state_db.write_run_cache(
            "test_run_cache_dir", deferred["run_cache_input_hash"], deferred["run_cache_entry"]
        )
        state_db.close()

    # Delete results and lock file to force run cache path
    # Lock file is in stages/ subdirectory
    shutil.rmtree(results_dir)
    lock_file = tmp_path / ".pivot" / "stages" / "test_run_cache_dir.lock"
    if lock_file.exists():
        lock_file.unlink()

    # Second run - should skip via run cache and restore directory
    result2 = executor.execute_stage("test_run_cache_dir", stage_info, worker_env, output_queue)
    assert result2["status"] == "skipped", f"Expected skipped, got {result2}"
    assert "run cache" in result2["reason"]

    # Verify files were restored
    assert results_dir.exists()
    assert (results_dir / "a.json").exists()
    assert (results_dir / "b.json").exists()

    # Verify content
    assert json.loads((results_dir / "a.json").read_text()) == {"value": 1}


def test_run_cache_skip_restores_corrupted_directory(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Run cache restores DirectoryOut when files are corrupted."""
    out_specs = stage_def.get_output_specs_from_return(_run_cache_directory_stage, "test_stage")
    dir_out: outputs.BaseOut = outputs.DirectoryOut(
        str(tmp_path / "results") + "/", loaders.JSON[dict[str, int]]()
    )

    stage_info = _make_stage_info(
        _run_cache_directory_stage,
        tmp_path,
        fingerprint={"self:_run_cache_directory_stage": "fp123"},
        out_specs=out_specs,
        outs=[dir_out],
    )

    # First run
    result1 = executor.execute_stage("test_run_cache_corrupt", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"

    results_dir = tmp_path / "results"

    # Manually write the run cache entry (normally done by coordinator)
    deferred = result1.get("deferred_writes", {})
    if "run_cache_input_hash" in deferred and "run_cache_entry" in deferred:
        state_db = state.StateDB(tmp_path / ".pivot" / "state.db")
        state_db.write_run_cache(
            "test_run_cache_corrupt", deferred["run_cache_input_hash"], deferred["run_cache_entry"]
        )
        state_db.close()

    # Delete lock file to force run cache path, but corrupt directory
    # Lock file is in stages/ subdirectory
    lock_file = tmp_path / ".pivot" / "stages" / "test_run_cache_corrupt.lock"
    if lock_file.exists():
        lock_file.unlink()

    # Corrupt a.json
    (results_dir / "a.json").unlink()
    (results_dir / "a.json").write_text('{"corrupted": true}')

    # Second run - should skip via run cache and fix corrupted file
    result2 = executor.execute_stage("test_run_cache_corrupt", stage_info, worker_env, output_queue)
    assert result2["status"] == "skipped"
    assert "run cache" in result2["reason"]

    # Verify corrupted file was restored
    assert json.loads((results_dir / "a.json").read_text()) == {"value": 1}


# =============================================================================
# Error Path Tests
# =============================================================================


def test_restore_outputs_fails_when_cache_missing(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """_restore_outputs returns False when cached file is missing."""
    files_cache_dir = tmp_path / ".pivot" / "cache" / "files"
    files_cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(tmp_path / "output.txt")
    # Hash must be exactly 16 characters (xxhash64 hexdigest)
    output_hash_map: dict[str, OutputHash] = {output_path: {"hash": "1234567890abcdef"}}

    result = worker._restore_outputs(
        [output_path],
        output_hash_map,
        files_cache_dir,
        [cache.CheckoutMode.COPY],
    )

    assert result is False, "Should fail when cache is missing"


def test_restore_outputs_returns_false_for_unrecorded_output(
    tmp_path: pathlib.Path,
) -> None:
    """_restore_outputs returns False when output is not in hash map."""
    files_cache_dir = tmp_path / ".pivot" / "cache" / "files"
    files_cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(tmp_path / "output.txt")
    output_hash_map: dict[str, OutputHash] = {}  # Empty map

    result = worker._restore_outputs(
        [output_path],
        output_hash_map,
        files_cache_dir,
        [cache.CheckoutMode.COPY],
    )

    assert result is False, "Should fail when output not recorded"


def test_restore_outputs_cleans_up_on_partial_failure(
    tmp_path: pathlib.Path,
) -> None:
    """_restore_outputs cleans up partially restored files on failure."""
    files_cache_dir = tmp_path / ".pivot" / "cache" / "files"
    files_cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a temp file and hash it to get a valid hash
    temp_file = tmp_path / "temp_for_hash.txt"
    temp_file.write_text("output1 content")
    output1_hash = cache.hash_file(temp_file)

    # Create cached file in the expected location
    cached_file = files_cache_dir / output1_hash[:2] / output1_hash[2:]
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    cached_file.write_text("output1 content")

    output1_path = str(tmp_path / "output1.txt")
    output2_path = str(tmp_path / "output2.txt")  # This won't have cache

    output_hash_map: dict[str, OutputHash] = {
        output1_path: {"hash": output1_hash},
        output2_path: {"hash": "fedcba0987654321"},  # Valid format but cache doesn't exist
    }

    result = worker._restore_outputs(
        [output1_path, output2_path],
        output_hash_map,
        files_cache_dir,
        [cache.CheckoutMode.COPY],
    )

    assert result is False, "Should fail when second output can't be restored"
    # First output should be cleaned up
    assert not pathlib.Path(output1_path).exists(), "Partially restored file should be cleaned up"


def test_try_skip_via_run_cache_returns_none_for_incremental_out(
    tmp_path: pathlib.Path,
) -> None:
    """_try_skip_via_run_cache returns None for IncrementalOut stages."""
    from pivot.storage import state

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    state_db = state.StateDB(state_dir / "state.db")

    incremental_out = outputs.IncrementalOut("output.txt", loaders.PathOnly())

    result = worker._try_skip_via_run_cache(
        "test_stage",
        "input_hash_123",
        [incremental_out],
        tmp_path / ".pivot" / "cache" / "files",
        [cache.CheckoutMode.COPY],
        state_db,
    )

    assert result is None, "IncrementalOut stages should not use run cache"
    state_db.close()


def test_try_skip_via_run_cache_returns_none_when_no_entry(
    tmp_path: pathlib.Path,
) -> None:
    """_try_skip_via_run_cache returns None when no run cache entry exists."""
    from pivot.storage import state

    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    state_db = state.StateDB(state_dir / "state.db")

    out = outputs.Out(str(tmp_path / "output.txt"), loaders.PathOnly())

    result = worker._try_skip_via_run_cache(
        "nonexistent_stage",
        "input_hash_never_seen",
        [out],
        tmp_path / ".pivot" / "cache" / "files",
        [cache.CheckoutMode.COPY],
        state_db,
    )

    assert result is None, "Should return None when no run cache entry"
    state_db.close()


def test_execute_stage_returns_failed_for_missing_directory_dep(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Worker returns failed status when directory dependency is missing."""

    def stage_func() -> None:
        pass

    stage_info = _make_stage_info(stage_func, tmp_path, deps=["missing_dir/"])
    result = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert result["status"] == "failed"
    assert "missing deps" in result["reason"]
    assert "missing_dir" in result["reason"]


# =============================================================================
# prepare_worker_info Tests
# =============================================================================


class _PrepareWorkerInfoOutput(TypedDict):
    """TypedDict for prepare_worker_info test stages."""

    result: Annotated[pathlib.Path, outputs.Out("result.txt", loaders.PathOnly())]


def _stage_with_custom_state_dir() -> _PrepareWorkerInfoOutput:
    """Stage function for testing prepare_worker_info with custom state_dir."""
    return _PrepareWorkerInfoOutput(result=pathlib.Path("result.txt"))


def _stage_without_custom_state_dir() -> _PrepareWorkerInfoOutput:
    """Stage function for testing prepare_worker_info with default state_dir."""
    return _PrepareWorkerInfoOutput(result=pathlib.Path("result.txt"))


def test_prepare_worker_info_uses_stage_state_dir(
    set_project_root: pathlib.Path,
    test_registry: registry.StageRegistry,
) -> None:
    """prepare_worker_info should use stage's state_dir when set."""
    custom_state_dir = set_project_root / "custom_pipeline" / ".pivot"
    test_registry.register(
        _stage_with_custom_state_dir,
        name="stage_with_custom_state",
        state_dir=custom_state_dir,
    )

    stage_info = test_registry.get("stage_with_custom_state")
    worker_info = executor_core.prepare_worker_info(
        stage_info=stage_info,
        overrides={},
        checkout_modes=[],
        run_id="test-run",
        force=False,
        no_commit=False,
        no_cache=False,
        project_root=set_project_root,
        default_state_dir=set_project_root / ".pivot",  # Fallback
    )

    assert worker_info["state_dir"] == custom_state_dir


def test_prepare_worker_info_uses_default_state_dir_when_stage_has_none(
    set_project_root: pathlib.Path,
    test_registry: registry.StageRegistry,
) -> None:
    """prepare_worker_info should use default_state_dir when stage has no state_dir."""
    # Register without state_dir
    test_registry.register(
        _stage_without_custom_state_dir,
        name="stage_without_custom_state",
    )

    stage_info = test_registry.get("stage_without_custom_state")
    default_state_dir = set_project_root / ".pivot"
    worker_info = executor_core.prepare_worker_info(
        stage_info=stage_info,
        overrides={},
        checkout_modes=[],
        run_id="test-run",
        force=False,
        no_commit=False,
        no_cache=False,
        project_root=set_project_root,
        default_state_dir=default_state_dir,
    )

    assert worker_info["state_dir"] == default_state_dir
