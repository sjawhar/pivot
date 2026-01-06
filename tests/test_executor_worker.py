"""Unit tests for executor worker functions.

These functions run in subprocesses during normal execution but are tested
directly here to avoid subprocess coverage issues.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from typing import TYPE_CHECKING, cast

import pytest

from pivot import exceptions, executor

if TYPE_CHECKING:
    import pathlib

    from pivot.executor import WorkerStageInfo
    from pivot.types import OutputMessage


@pytest.fixture
def worker_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up worker execution environment."""
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return cache_dir


# =============================================================================
# execute_stage_worker Tests
# =============================================================================


def test_execute_stage_worker_with_missing_deps(worker_env: pathlib.Path) -> None:
    """Worker returns failed status when dependency files are missing."""
    stage_info: WorkerStageInfo = {
        "func": lambda: None,
        "fingerprint": {"self:test": "abc123"},
        "deps": ["missing_file.txt"],
        "signature": None,
        "outs": [],
    }

    result = executor.execute_stage_worker(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "failed"
    assert "missing deps" in result["reason"]
    assert "missing_file.txt" in result["reason"]


def test_execute_stage_worker_with_directory_dep(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Worker hashes directory dependency and runs stage."""
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    (data_dir / "file.txt").write_text("content")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("done")

    stage_info: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:test": "abc123"},
        "deps": ["data_dir"],
        "signature": None,
        "outs": [],
    }

    result = executor.execute_stage_worker(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "ran", f"Expected ran, got {result}"
    assert (tmp_path / "output.txt").read_text() == "done"


def test_execute_stage_worker_runs_unchanged_stage(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Worker skips stage when fingerprint matches and deps unchanged."""
    (tmp_path / "input.txt").write_text("data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("result")

    stage_info: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:stage_func": "fp123"},
        "deps": ["input.txt"],
        "signature": None,
        "outs": [],
    }

    # First run - creates lock file
    result1 = executor.execute_stage_worker(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result1["status"] == "ran"
    assert (tmp_path / "output.txt").read_text() == "result"

    # Second run - should skip (unchanged)
    result2 = executor.execute_stage_worker(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result2["status"] == "skipped"
    assert result2["reason"] == "unchanged"


def test_execute_stage_worker_reruns_when_fingerprint_changes(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Worker reruns stage when code fingerprint changes."""
    (tmp_path / "input.txt").write_text("data")
    counter = tmp_path / "counter.txt"

    def stage_func_v1() -> None:
        count = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(count + 1))

    stage_info_v1: WorkerStageInfo = {
        "func": stage_func_v1,
        "fingerprint": {"self:stage_func_v1": "fp_v1"},
        "deps": ["input.txt"],
        "signature": None,
        "outs": [],
    }

    # First run
    result1 = executor.execute_stage_worker(
        "test_stage",
        stage_info_v1,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result1["status"] == "ran"
    assert counter.read_text() == "1"

    # Second run with different fingerprint
    stage_info_v2: WorkerStageInfo = {
        **stage_info_v1,
        "fingerprint": {"self:stage_func_v1": "fp_v2"},
    }
    result2 = executor.execute_stage_worker(
        "test_stage",
        stage_info_v2,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result2["status"] == "ran"
    assert result2["reason"] == "Code changed"
    assert counter.read_text() == "2"


def test_execute_stage_worker_handles_stage_exception(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Worker returns failed status when stage raises exception."""
    (tmp_path / "input.txt").write_text("data")

    def failing_stage() -> None:
        raise RuntimeError("Stage failed intentionally")

    stage_info: WorkerStageInfo = {
        "func": failing_stage,
        "fingerprint": {"self:failing_stage": "fp123"},
        "deps": ["input.txt"],
        "signature": None,
        "outs": [],
    }

    result = executor.execute_stage_worker(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "failed"
    assert "Stage failed intentionally" in result["reason"]


def test_execute_stage_worker_handles_sys_exit(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Worker catches sys.exit and returns failed status."""
    (tmp_path / "input.txt").write_text("data")

    def exits_stage() -> None:
        sys.exit(42)

    stage_info: WorkerStageInfo = {
        "func": exits_stage,
        "fingerprint": {"self:exits_stage": "fp123"},
        "deps": ["input.txt"],
        "signature": None,
        "outs": [],
    }

    result = executor.execute_stage_worker(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "failed"
    assert "sys.exit" in result["reason"]
    assert "42" in result["reason"]


def test_execute_stage_worker_handles_keyboard_interrupt(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Worker catches KeyboardInterrupt and returns failed status."""
    (tmp_path / "input.txt").write_text("data")

    def interrupted_stage() -> None:
        raise KeyboardInterrupt("User cancelled")

    stage_info: WorkerStageInfo = {
        "func": interrupted_stage,
        "fingerprint": {"self:interrupted_stage": "fp123"},
        "deps": ["input.txt"],
        "signature": None,
        "outs": [],
    }

    result = executor.execute_stage_worker(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "failed"
    assert "KeyboardInterrupt" in result["reason"] or "User cancelled" in result["reason"]


# =============================================================================
# _run_stage_function_with_capture Tests
# =============================================================================


def test_run_stage_function_captures_stdout() -> None:
    """Captures stdout from stage function."""

    def stage_with_output() -> None:
        print("line1")
        print("line2")

    output_lines = executor._run_stage_function_with_capture(
        stage_with_output,
        "test_stage",
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert len(output_lines) == 2
    assert output_lines[0] == ("line1", False)  # stdout
    assert output_lines[1] == ("line2", False)


def test_run_stage_function_captures_stderr() -> None:
    """Captures stderr from stage function."""

    def stage_with_errors() -> None:
        print("error1", file=sys.stderr)
        print("error2", file=sys.stderr)

    output_lines = executor._run_stage_function_with_capture(
        stage_with_errors,
        "test_stage",
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert len(output_lines) == 2
    assert output_lines[0] == ("error1", True)  # stderr
    assert output_lines[1] == ("error2", True)


def test_run_stage_function_captures_mixed_output() -> None:
    """Captures both stdout and stderr."""

    def stage_mixed() -> None:
        print("stdout1")
        print("stderr1", file=sys.stderr)
        print("stdout2")

    output_lines = executor._run_stage_function_with_capture(
        stage_mixed,
        "test_stage",
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert len(output_lines) == 3
    assert output_lines[0] == ("stdout1", False)
    assert output_lines[1] == ("stderr1", True)
    assert output_lines[2] == ("stdout2", False)


def test_run_stage_function_restores_streams() -> None:
    """Restores original stdout/stderr after execution."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    def noop_stage() -> None:
        pass

    executor._run_stage_function_with_capture(noop_stage, "test", mp.Manager().Queue())  # pyright: ignore[reportArgumentType]

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


def test_run_stage_function_restores_streams_on_exception() -> None:
    """Restores streams even when stage raises exception."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    def failing_stage() -> None:
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        executor._run_stage_function_with_capture(failing_stage, "test", mp.Manager().Queue())  # pyright: ignore[reportArgumentType]

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


def test_run_stage_function_captures_partial_lines() -> None:
    """Captures output without trailing newline."""

    def stage_no_newline() -> None:
        sys.stdout.write("no newline")
        sys.stdout.flush()

    output_lines = executor._run_stage_function_with_capture(
        stage_no_newline,
        "test_stage",
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert len(output_lines) == 1
    assert output_lines[0] == ("no newline", False)


# =============================================================================
# _QueueStreamCapture Tests
# =============================================================================


def test_queue_stream_capture_splits_on_newlines() -> None:
    """QueueStreamCapture splits output on newlines."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()  # pyright: ignore[reportAssignmentType]

    capture = executor._QueueStreamCapture(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    bytes_written = capture.write("line1\nline2\n")

    assert bytes_written == len("line1\nline2\n")
    assert output_lines == [("line1", False), ("line2", False)]


def test_queue_stream_capture_buffers_partial_lines() -> None:
    """QueueStreamCapture buffers incomplete lines."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()  # pyright: ignore[reportAssignmentType]

    capture = executor._QueueStreamCapture(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    capture.write("partial")
    assert output_lines == []  # Not flushed yet

    capture.write(" line\n")
    assert output_lines == [("partial line", False)]


def test_queue_stream_capture_flush_writes_buffer() -> None:
    """QueueStreamCapture.flush() writes buffered content."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()  # pyright: ignore[reportAssignmentType]

    capture = executor._QueueStreamCapture(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    capture.write("no newline")
    assert output_lines == []

    capture.flush()
    assert output_lines == [("no newline", False)]


def test_queue_stream_capture_distinguishes_stderr() -> None:
    """QueueStreamCapture marks stderr lines correctly."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()  # pyright: ignore[reportAssignmentType]

    capture = executor._QueueStreamCapture(
        "test_stage",
        queue,
        is_stderr=True,
        output_lines=output_lines,
    )

    capture.write("error\n")
    assert output_lines == [("error", True)]


def test_queue_stream_capture_handles_multiple_newlines() -> None:
    """QueueStreamCapture handles text with multiple consecutive newlines."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()  # pyright: ignore[reportAssignmentType]

    capture = executor._QueueStreamCapture(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    capture.write("line1\n\nline2\n")
    # Empty lines are skipped (code checks 'if line:')
    assert output_lines == [("line1", False), ("line2", False)]


def test_queue_stream_capture_empty_flush_does_nothing() -> None:
    """QueueStreamCapture.flush() with empty buffer does nothing."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()  # pyright: ignore[reportAssignmentType]

    capture = executor._QueueStreamCapture(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    capture.flush()
    assert output_lines == []


def test_queue_stream_capture_textio_protocol() -> None:
    """QueueStreamCapture implements TextIO protocol properties."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()  # pyright: ignore[reportAssignmentType]

    capture = executor._QueueStreamCapture(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    # TextIO protocol properties
    assert capture.encoding == "utf-8"
    assert capture.errors == "strict"
    assert capture.newlines is None

    # Stream capability properties
    assert capture.readable() is False
    assert capture.writable() is True
    assert capture.seekable() is False
    assert capture.isatty() is False

    # fileno raises OSError (no underlying fd)
    import pytest

    with pytest.raises(OSError, match="no underlying file descriptor"):
        capture.fileno()


# =============================================================================
# Execution Lock Tests
# =============================================================================


def test_execution_lock_creates_sentinel_file(worker_env: pathlib.Path) -> None:
    """Execution lock creates sentinel file during execution."""
    sentinel_path = worker_env / "test_stage.running"

    with executor._execution_lock("test_stage", worker_env) as sentinel:
        assert sentinel.exists()
        assert sentinel == sentinel_path
        content = sentinel.read_text()
        assert "pid:" in content

    # Cleaned up after context
    assert not sentinel.exists()


def test_execution_lock_removes_sentinel_on_exception(worker_env: pathlib.Path) -> None:
    """Execution lock removes sentinel even when exception occurs."""
    sentinel_path = worker_env / "test_stage.running"

    with pytest.raises(RuntimeError), executor._execution_lock("test_stage", worker_env):
        assert sentinel_path.exists()
        raise RuntimeError("intentional")

    assert not sentinel_path.exists()


def test_acquire_execution_lock_succeeds_when_available(worker_env: pathlib.Path) -> None:
    """Acquire lock succeeds when no lock exists."""
    sentinel = executor._acquire_execution_lock("test_stage", worker_env)

    assert sentinel.exists()
    assert sentinel == worker_env / "test_stage.running"

    # Cleanup
    sentinel.unlink()


def test_acquire_execution_lock_fails_when_held_by_live_process(
    worker_env: pathlib.Path,
) -> None:
    """Acquire lock fails when held by a running process."""
    sentinel = worker_env / "test_stage.running"
    sentinel.write_text(f"pid: {os.getpid()}\n")

    with pytest.raises(exceptions.StageAlreadyRunningError) as exc_info:
        executor._acquire_execution_lock("test_stage", worker_env)

    assert "already running" in str(exc_info.value)
    assert str(os.getpid()) in str(exc_info.value)

    # Cleanup
    sentinel.unlink()


def test_acquire_execution_lock_breaks_stale_lock(worker_env: pathlib.Path) -> None:
    """Acquire lock breaks stale lock from dead process."""
    sentinel = worker_env / "test_stage.running"
    sentinel.write_text("pid: 999999999\n")  # Non-existent PID

    result_sentinel = executor._acquire_execution_lock("test_stage", worker_env)

    assert result_sentinel.exists()
    assert result_sentinel == sentinel

    # Cleanup
    result_sentinel.unlink()


def test_acquire_execution_lock_breaks_corrupted_lock(worker_env: pathlib.Path) -> None:
    """Acquire lock breaks corrupted lock file."""
    sentinel = worker_env / "test_stage.running"
    sentinel.write_text("corrupted content")

    result_sentinel = executor._acquire_execution_lock("test_stage", worker_env)

    assert result_sentinel.exists()

    # Cleanup
    result_sentinel.unlink()


def test_acquire_execution_lock_breaks_negative_pid_lock(worker_env: pathlib.Path) -> None:
    """Acquire lock breaks lock with invalid negative PID."""
    sentinel = worker_env / "test_stage.running"
    sentinel.write_text("pid: -1\n")

    result_sentinel = executor._acquire_execution_lock("test_stage", worker_env)

    assert result_sentinel.exists()

    # Cleanup
    result_sentinel.unlink()


# =============================================================================
# Process Alive Check Tests
# =============================================================================


def test_is_process_alive_returns_true_for_self() -> None:
    """is_process_alive returns True for own PID."""
    assert executor._is_process_alive(os.getpid())


def test_is_process_alive_returns_false_for_nonexistent() -> None:
    """is_process_alive returns False for non-existent PID."""
    assert not executor._is_process_alive(999999999)


def test_is_process_alive_returns_true_for_init() -> None:
    """is_process_alive returns True for PID 1 (init/systemd)."""
    # PID 1 always exists (init/systemd)
    assert executor._is_process_alive(1)


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_extract_params_with_no_signature() -> None:
    """extract_params returns empty dict when no signature."""
    stage_info = cast("WorkerStageInfo", cast("object", {"signature": None}))
    params = executor.extract_params(stage_info)
    assert params == {}


def test_extract_params_with_no_defaults() -> None:
    """extract_params returns empty dict when no default parameters."""
    import inspect

    def func(a: int, b: str) -> None:
        pass

    stage_info = cast("WorkerStageInfo", cast("object", {"signature": inspect.signature(func)}))
    params = executor.extract_params(stage_info)
    assert params == {}


def test_extract_params_with_defaults() -> None:
    """extract_params extracts parameters with default values."""
    import inspect

    def func(a: int, b: str = "default", c: float = 3.14) -> None:
        pass

    stage_info = cast("WorkerStageInfo", cast("object", {"signature": inspect.signature(func)}))
    params = executor.extract_params(stage_info)
    assert params == {"b": "default", "c": 3.14}


def test_hash_dependencies_with_existing_files(tmp_path: pathlib.Path) -> None:
    """hash_dependencies hashes existing files."""
    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")

    import os

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        hashes, missing = executor.hash_dependencies(["file1.txt", "file2.txt"])

        assert len(hashes) == 2
        assert "file1.txt" in hashes
        assert "file2.txt" in hashes
        assert len(missing) == 0
    finally:
        os.chdir(old_cwd)


def test_hash_dependencies_with_missing_files() -> None:
    """hash_dependencies reports missing files."""
    hashes, missing = executor.hash_dependencies(["missing1.txt", "missing2.txt"])

    assert len(hashes) == 0
    assert missing == ["missing1.txt", "missing2.txt"]


def test_hash_dependencies_with_directory(tmp_path: pathlib.Path) -> None:
    """hash_dependencies hashes directories using tree hash."""
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    (data_dir / "file.txt").write_text("content")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        hashes, missing = executor.hash_dependencies(["data_dir"])

        assert len(hashes) == 1, "Directory should be hashed"
        assert "data_dir" in hashes
        assert len(missing) == 0, "No missing dependencies"
    finally:
        os.chdir(old_cwd)


def test_hash_file_produces_consistent_hash(tmp_path: pathlib.Path) -> None:
    """hash_file produces same hash for same content."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")

    hash1 = executor.hash_file(file_path)
    hash2 = executor.hash_file(file_path)

    assert hash1 == hash2
    assert len(hash1) == 16  # First 16 hex chars of SHA256


def test_hash_file_different_for_different_content(tmp_path: pathlib.Path) -> None:
    """hash_file produces different hash for different content."""
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("content1")
    file2.write_text("content2")

    hash1 = executor.hash_file(file1)
    hash2 = executor.hash_file(file2)

    assert hash1 != hash2
