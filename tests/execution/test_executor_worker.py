# pyright: reportAssignmentType=false, reportIncompatibleVariableOverride=false, reportAttributeAccessIssue=false
from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import os
import pathlib
import sys
from typing import TYPE_CHECKING, Any

import pandas  # noqa: TC002 - needed at runtime for StageDef type hint resolution
import pydantic
import pytest

from pivot import exceptions, executor, loaders, outputs, stage_def
from pivot.executor import worker
from pivot.storage import cache, lock

if TYPE_CHECKING:
    from collections.abc import Generator

    from pivot.executor import WorkerStageInfo
    from pivot.types import DirManifestEntry, OutputMessage


# Module-level StageDef classes for testing (required for type hint resolution)
class _TestStageDef_Deps(stage_def.StageDef):
    """StageDef with deps for testing."""

    class deps:
        data: loaders.CSV[pandas.DataFrame] = "input.csv"


class _TestStageDef_Outs(stage_def.StageDef):
    """StageDef with outs for testing."""

    class outs:
        result: loaders.JSON[dict[str, int]] = "output.json"


class _TestStageDef_MissingDeps(stage_def.StageDef):
    """StageDef with deps that will be missing."""

    class deps:
        data: loaders.CSV[pandas.DataFrame] = "missing.csv"


class _PlainParams(pydantic.BaseModel):
    """Plain Pydantic params for testing backward compatibility."""

    threshold: float = 0.5


@pytest.fixture
def worker_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up worker execution environment."""
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return cache_dir


def _helper_always_fail_takeover(sentinel: pathlib.Path, stale_pid: int | None) -> bool:
    """Helper that always fails lock takeover (for testing retry exhaustion)."""
    _ = sentinel, stale_pid  # Unused
    return False


# =============================================================================
# execute_stage Tests
# =============================================================================


def test_execute_stage_with_missing_deps(worker_env: pathlib.Path) -> None:
    """Worker returns failed status when dependency files are missing."""
    stage_info: WorkerStageInfo = {
        "func": lambda: None,
        "fingerprint": {"self:test": "abc123"},
        "deps": ["missing_file.txt"],
        "signature": None,
        "outs": [],
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "failed"
    assert "missing deps" in result["reason"]
    assert "missing_file.txt" in result["reason"]


def test_execute_stage_with_directory_dep(worker_env: pathlib.Path, tmp_path: pathlib.Path) -> None:
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
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "ran", f"Expected ran, got {result}"
    assert (tmp_path / "output.txt").read_text() == "done"


def test_execute_stage_runs_unchanged_stage(
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
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    # First run - creates lock file
    result1 = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result1["status"] == "ran"
    assert (tmp_path / "output.txt").read_text() == "result"

    # Second run - should skip (unchanged)
    result2 = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result2["status"] == "skipped"
    assert result2["reason"] == "unchanged"


def test_execute_stage_reruns_when_fingerprint_changes(
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
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    # First run
    result1 = executor.execute_stage(
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
    result2 = executor.execute_stage(
        "test_stage",
        stage_info_v2,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result2["status"] == "ran"
    assert result2["reason"] == "Code changed"
    assert counter.read_text() == "2"


def test_execute_stage_handles_stage_exception(
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
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "failed"
    assert "Stage failed intentionally" in result["reason"]


def test_execute_stage_handles_sys_exit(worker_env: pathlib.Path, tmp_path: pathlib.Path) -> None:
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
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "failed"
    assert "sys.exit" in result["reason"]
    assert "42" in result["reason"]


def test_execute_stage_handles_keyboard_interrupt(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Worker returns failed status for KeyboardInterrupt."""
    (tmp_path / "input.txt").write_text("data")

    def interrupted_stage() -> None:
        raise KeyboardInterrupt("User cancelled")

    stage_info: WorkerStageInfo = {
        "func": interrupted_stage,
        "fingerprint": {"self:interrupted_stage": "fp123"},
        "deps": ["input.txt"],
        "signature": None,
        "outs": [],
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "failed"
    assert "KeyboardInterrupt" in result["reason"]


# =============================================================================
# _run_stage_function_with_capture Tests
# =============================================================================


def test_run_stage_function_captures_stdout() -> None:
    """Captures stdout from stage function."""

    def stage_with_output() -> None:
        print("line1")
        print("line2")

    output_lines: list[tuple[str, bool]] = []
    worker._run_stage_function_with_capture(
        stage_with_output,
        "test_stage",
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
        output_lines,
    )

    assert len(output_lines) == 2
    assert output_lines[0] == ("line1", False)  # stdout
    assert output_lines[1] == ("line2", False)


def test_run_stage_function_captures_stderr() -> None:
    """Captures stderr from stage function."""

    def stage_with_errors() -> None:
        print("error1", file=sys.stderr)
        print("error2", file=sys.stderr)

    output_lines: list[tuple[str, bool]] = []
    worker._run_stage_function_with_capture(
        stage_with_errors,
        "test_stage",
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
        output_lines,
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

    output_lines: list[tuple[str, bool]] = []
    worker._run_stage_function_with_capture(
        stage_mixed,
        "test_stage",
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
        output_lines,
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

    output_lines: list[tuple[str, bool]] = []
    worker._run_stage_function_with_capture(
        noop_stage,
        "test",
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
        output_lines,
    )

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


def test_run_stage_function_restores_streams_on_exception() -> None:
    """Restores streams even when stage raises exception."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    def failing_stage() -> None:
        raise RuntimeError("fail")

    output_lines: list[tuple[str, bool]] = []
    with pytest.raises(RuntimeError):
        worker._run_stage_function_with_capture(
            failing_stage,
            "test",
            mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
            output_lines,
        )

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


def test_run_stage_function_captures_partial_lines() -> None:
    """Captures output without trailing newline."""

    def stage_no_newline() -> None:
        sys.stdout.write("no newline")
        sys.stdout.flush()

    output_lines: list[tuple[str, bool]] = []
    worker._run_stage_function_with_capture(
        stage_no_newline,
        "test_stage",
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
        output_lines,
    )

    assert len(output_lines) == 1
    assert output_lines[0] == ("no newline", False)


# =============================================================================
# _QueueWriter Tests
# =============================================================================


def test_queue_writer_splits_on_newlines() -> None:
    """_QueueWriter splits output on newlines."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()

    writer = worker._QueueWriter(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    bytes_written = writer.write("line1\nline2\n")

    assert bytes_written == len("line1\nline2\n")
    assert output_lines == [("line1", False), ("line2", False)]


def test_queue_writer_buffers_partial_lines() -> None:
    """_QueueWriter buffers incomplete lines."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()

    writer = worker._QueueWriter(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    writer.write("partial")
    assert output_lines == []  # Not flushed yet

    writer.write(" line\n")
    assert output_lines == [("partial line", False)]


def test_queue_writer_flush_writes_buffer() -> None:
    """_QueueWriter.flush() writes buffered content."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()

    writer = worker._QueueWriter(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    writer.write("no newline")
    assert output_lines == []

    writer.flush()
    assert output_lines == [("no newline", False)]


def test_queue_writer_distinguishes_stderr() -> None:
    """_QueueWriter marks stderr lines correctly."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()

    writer = worker._QueueWriter(
        "test_stage",
        queue,
        is_stderr=True,
        output_lines=output_lines,
    )

    writer.write("error\n")
    assert output_lines == [("error", True)]


def test_queue_writer_handles_multiple_newlines() -> None:
    """_QueueWriter handles text with multiple consecutive newlines."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()

    writer = worker._QueueWriter(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    writer.write("line1\n\nline2\n")
    # Empty lines are skipped (code checks 'if line:')
    assert output_lines == [("line1", False), ("line2", False)]


def test_queue_writer_empty_flush_does_nothing() -> None:
    """_QueueWriter.flush() with empty buffer does nothing."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()

    writer = worker._QueueWriter(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    writer.flush()
    assert output_lines == []


def test_queue_writer_isatty_returns_false() -> None:
    """_QueueWriter.isatty() returns False."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()

    writer = worker._QueueWriter(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    assert writer.isatty() is False


def test_queue_writer_fileno_raises_unsupported_operation() -> None:
    """_QueueWriter.fileno() raises io.UnsupportedOperation."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()

    writer = worker._QueueWriter(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    )

    with pytest.raises(io.UnsupportedOperation, match="file descriptor"):
        writer.fileno()


def test_queue_writer_context_manager_flushes_on_exit() -> None:
    """_QueueWriter context manager flushes buffer on exit."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()

    with worker._QueueWriter(
        "test_stage",
        queue,
        is_stderr=False,
        output_lines=output_lines,
    ) as writer:
        writer.write("no newline")
        assert output_lines == []  # Not flushed yet

    # Flushed on context exit
    assert output_lines == [("no newline", False)]


def test_queue_writer_context_manager_flushes_on_exception() -> None:
    """_QueueWriter context manager flushes buffer even when exception raised."""
    output_lines: list[tuple[str, bool]] = []
    queue: mp.Queue[OutputMessage] = mp.Manager().Queue()

    with (
        pytest.raises(RuntimeError),
        worker._QueueWriter(
            "test_stage",
            queue,
            is_stderr=False,
            output_lines=output_lines,
        ),
    ):
        print("before error")
        raise RuntimeError("test error")

    # Output captured despite exception
    assert output_lines == [("before error", False)]


def test_run_stage_function_preserves_output_on_exception() -> None:
    """Output is preserved even when stage function raises exception."""

    def failing_stage() -> None:
        print("line before error")
        raise RuntimeError("stage failed")

    output_lines: list[tuple[str, bool]] = []
    with pytest.raises(RuntimeError):
        worker._run_stage_function_with_capture(
            failing_stage,
            "test_stage",
            mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
            output_lines,
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
        assert "pid:" in content

    # Cleaned up after context
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
    sentinel.write_text(f"pid: {os.getpid()}\n")

    with pytest.raises(exceptions.StageAlreadyRunningError) as exc_info:
        lock.acquire_execution_lock("test_stage", worker_env)

    assert "already running" in str(exc_info.value)
    assert str(os.getpid()) in str(exc_info.value)

    # Cleanup
    sentinel.unlink()


def test_acquire_execution_lock_breaks_stale_lock(worker_env: pathlib.Path) -> None:
    """Acquire lock breaks stale lock from dead process."""
    sentinel = worker_env / "test_stage.running"
    sentinel.write_text("pid: 999999999\n")  # Non-existent PID

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
    sentinel.write_text("pid: -1\n")

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
    sentinel.write_text("pid: 12345\n")

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
    sentinel.write_text("pid: -1\n")

    assert lock._read_lock_pid(sentinel) is None


def test_read_lock_pid_returns_none_for_zero_pid(worker_env: pathlib.Path) -> None:
    """_read_lock_pid returns None for invalid zero PID."""
    sentinel = worker_env / "test.running"
    sentinel.write_text("pid: 0\n")

    assert lock._read_lock_pid(sentinel) is None


# =============================================================================
# _atomic_lock_takeover Tests
# =============================================================================


def test_atomic_lock_takeover_succeeds_on_stale_lock(worker_env: pathlib.Path) -> None:
    """Atomic takeover creates lock with current process PID."""
    sentinel = worker_env / "test_stage.running"
    sentinel.write_text("pid: 999999999\n")  # Stale lock

    result = lock._atomic_lock_takeover(sentinel, 999999999)

    assert result is True
    assert sentinel.exists()
    content = sentinel.read_text()
    assert f"pid: {os.getpid()}" in content

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
        pathlib.Path(dst).write_text("pid: 999888777\n")

    monkeypatch.setattr(os, "replace", sneaky_replace)
    sentinel.write_text("pid: 999999999\n")  # Stale lock

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
    sentinel.write_text("pid: 999999999\n")

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
        sent.write_text(f"pid: {my_pid}\n")
        return True

    # Start with stale lock
    sentinel.write_text("pid: 999999999\n")
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
    sentinel.write_text("pid: 999999999\n")

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
    stale_sentinel.write_text("pid: 999999999\n")

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


def test_hash_dependencies_with_existing_files(tmp_path: pathlib.Path) -> None:
    """hash_dependencies hashes existing files as FileHash dicts."""
    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
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
    finally:
        os.chdir(old_cwd)


def test_hash_dependencies_with_missing_files() -> None:
    """hash_dependencies reports missing files."""
    hashes, missing, unreadable = executor.hash_dependencies(["missing1.txt", "missing2.txt"])

    assert len(hashes) == 0
    assert missing == ["missing1.txt", "missing2.txt"]
    assert len(unreadable) == 0


def test_hash_dependencies_with_directory(tmp_path: pathlib.Path) -> None:
    """hash_dependencies hashes directories with manifest."""
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    (data_dir / "file.txt").write_text("content")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
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
    finally:
        os.chdir(old_cwd)


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


def test_generation_skip_on_second_run(worker_env: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Second run uses generation-based skip detection."""

    (tmp_path / "input.txt").write_text("data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("result")

    out = outputs.Out(str(tmp_path / "output.txt"))
    stage_info: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:stage_func": "fp123"},
        "deps": ["input.txt"],
        "signature": None,
        "outs": [out],
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    # First run - creates output and records generations
    result1 = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result1["status"] == "ran"
    assert (tmp_path / "output.txt").read_text() == "result"

    # Second run - should skip via generation check
    result2 = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result2["status"] == "skipped"
    # Falls back to hash-based skip because input.txt is external (no generation tracking)
    assert "unchanged" in result2["reason"]


def test_generation_mismatch_triggers_rerun(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
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

    step1_out = outputs.Out(str(tmp_path / "intermediate.txt"))
    step2_out = outputs.Out(str(tmp_path / "final.txt"))

    step1_info: WorkerStageInfo = {
        "func": step1_func,
        "fingerprint": {"self:step1": "fp1"},
        "deps": ["input.txt"],
        "signature": None,
        "outs": [step1_out],
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    step2_info: WorkerStageInfo = {
        "func": step2_func,
        "fingerprint": {"self:step2": "fp2"},
        "deps": ["intermediate.txt"],
        "signature": None,
        "outs": [step2_out],
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    # First run - both stages execute
    result1_step1 = executor.execute_stage(
        "step1",
        step1_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result1_step1["status"] == "ran"
    result1_step2 = executor.execute_stage(
        "step2",
        step2_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result1_step2["status"] == "ran"
    assert (tmp_path / "final.txt").read_text() == "Final: ORIGINAL"

    # Second run - both should skip
    result2_step1 = executor.execute_stage(
        "step1",
        step1_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result2_step1["status"] == "skipped"
    result2_step2 = executor.execute_stage(
        "step2",
        step2_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result2_step2["status"] == "skipped"

    # Change input - step1 should re-run
    (tmp_path / "input.txt").write_text("modified")
    result3_step1 = executor.execute_stage(
        "step1",
        step1_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result3_step1["status"] == "ran"

    # step2 should re-run because intermediate.txt generation changed
    result3_step2 = executor.execute_stage(
        "step2",
        step2_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result3_step2["status"] == "ran"
    assert (tmp_path / "final.txt").read_text() == "Final: MODIFIED"


def test_external_file_fallback_to_hash_check(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """External files (no generation) trigger fallback to hash-based check."""

    # Create external input file (not a Pivot output, so no generation)
    (tmp_path / "external_data.txt").write_text("external")

    def stage_func() -> None:
        data = (tmp_path / "external_data.txt").read_text()
        (tmp_path / "output.txt").write_text(data.upper())

    out = outputs.Out(str(tmp_path / "output.txt"))
    stage_info: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:stage_func": "fp123"},
        "deps": ["external_data.txt"],
        "signature": None,
        "outs": [out],
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    # First run
    result1 = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result1["status"] == "ran"
    assert (tmp_path / "output.txt").read_text() == "EXTERNAL"

    # Second run - should skip (external file has no generation, falls back to hash)
    result2 = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result2["status"] == "skipped"

    # Modify external file - should detect change via hash fallback
    (tmp_path / "external_data.txt").write_text("changed")
    result3 = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result3["status"] == "ran"
    assert (tmp_path / "output.txt").read_text() == "CHANGED"


def test_deps_list_change_triggers_rerun(worker_env: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Changing deps list (even with same fingerprint) triggers re-run via hash check.

    Generation tracking only checks current deps, so removing a dep from the list
    could cause incorrect skips. This is mitigated because:
    1. In real usage, deps come from @stage decorator which affects fingerprint
    2. The hash-based fallback compares full dep_hashes dict which catches changes

    This test verifies the hash-based fallback catches deps list changes.
    """
    (tmp_path / "dep_a.txt").write_text("A")
    (tmp_path / "dep_b.txt").write_text("B")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("done")

    out = outputs.Out(str(tmp_path / "output.txt"))

    # First run with deps=[A, B]
    stage_info_v1: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:stage": "fp1"},
        "deps": [str(tmp_path / "dep_a.txt"), str(tmp_path / "dep_b.txt")],
        "signature": None,
        "outs": [out],
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result1 = executor.execute_stage(
        "test_stage",
        stage_info_v1,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result1["status"] == "ran"

    # Second run with same config - should skip
    result2 = executor.execute_stage(
        "test_stage",
        stage_info_v1,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result2["status"] == "skipped"

    # Third run with deps=[A] only (B removed), DIFFERENT fingerprint
    # This simulates real usage where changing @stage(deps=...) changes fingerprint
    stage_info_v2: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:stage": "fp2"},  # Different fingerprint
        "deps": [str(tmp_path / "dep_a.txt")],  # B removed
        "signature": None,
        "outs": [out],
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result3 = executor.execute_stage(
        "test_stage",
        stage_info_v2,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result3["status"] == "ran", "Fingerprint change should trigger re-run"


def test_deps_list_change_same_fingerprint_detected_by_hash(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Even with same fingerprint, deps list change is caught by hash comparison.

    This is a safety test for the edge case where fingerprint somehow stays same
    but deps list changes. The hash-based fallback should catch this.
    """
    (tmp_path / "dep_a.txt").write_text("A")
    (tmp_path / "dep_b.txt").write_text("B")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("done")

    out = outputs.Out(str(tmp_path / "output.txt"))

    # First run with deps=[A, B]
    stage_info_v1: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:stage": "fp_same"},
        "deps": [str(tmp_path / "dep_a.txt"), str(tmp_path / "dep_b.txt")],
        "signature": None,
        "outs": [out],
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result1 = executor.execute_stage(
        "test_stage",
        stage_info_v1,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result1["status"] == "ran"

    # Second run with deps=[A] only (B removed), SAME fingerprint
    # Generation tracking would miss this, but hash comparison catches it
    stage_info_v2: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:stage": "fp_same"},  # Same fingerprint!
        "deps": [str(tmp_path / "dep_a.txt")],  # B removed
        "signature": None,
        "outs": [out],
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result2 = executor.execute_stage(
        "test_stage",
        stage_info_v2,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
    assert result2["status"] == "ran", (
        "Deps list change should trigger re-run even with same fingerprint"
    )


# =============================================================================
# TOCTOU Prevention Tests
# =============================================================================


def test_skip_acquires_execution_lock(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipped stages still acquire execution lock (TOCTOU prevention).

    This ensures output restoration happens inside the lock, preventing race
    conditions between parallel processes.
    """
    (tmp_path / "input.txt").write_text("data")

    def stage_func() -> None:
        (tmp_path / "output.txt").write_text("result")

    out = outputs.Out(str(tmp_path / "output.txt"))
    stage_info: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:stage_func": "fp123"},
        "deps": ["input.txt"],
        "signature": None,
        "outs": [out],
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    # First run - creates lock file and output
    result1 = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
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
    result2 = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result2["status"] == "skipped"
    assert lock_acquired, "Execution lock should be acquired even when skipping (TOCTOU prevention)"


def test_restore_happens_inside_lock(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output restoration occurs while execution lock is held.

    Verifies the fix for TOCTOU race condition where output could be modified
    between skip decision and restoration.
    """
    (tmp_path / "input.txt").write_text("data")
    output_path = tmp_path / "output.txt"

    def stage_func() -> None:
        output_path.write_text("result")

    out = outputs.Out(str(output_path))
    stage_info: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:stage_func": "fp123"},
        "deps": ["input.txt"],
        "signature": None,
        "outs": [out],
        "params": None,
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["copy"],  # Use copy mode for simpler testing
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    # First run - creates lock file and caches output
    result1 = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )
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
    result2 = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result2["status"] == "skipped"
    assert output_path.exists(), "Output should be restored"

    # Verify restore happened between lock acquire and release
    assert operations == ["lock_acquire", "restore", "lock_release"], (
        f"Restore should happen inside lock. Got: {operations}"
    )


# =============================================================================
# StageDef auto-load/save tests
# =============================================================================


def test_stage_def_deps_loaded_before_function(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """StageDef deps should be auto-loaded before the stage function runs."""
    input_file = tmp_path / "input.csv"
    input_file.write_text("a,b\n1,2\n3,4\n")

    loaded_data: list[Any] = []

    def stage_func(params: _TestStageDef_Deps) -> None:
        loaded_data.append(params.deps.data)

    stage_info: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:test": "abc123"},
        "deps": [str(input_file)],
        "signature": None,
        "outs": [],
        "params": _TestStageDef_Deps(),
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "ran"
    assert len(loaded_data) == 1
    assert hasattr(loaded_data[0], "columns")  # DataFrame


def test_stage_def_outs_saved_after_function(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """StageDef outs should be auto-saved after the stage function returns."""
    output_file = tmp_path / "output.json"

    def stage_func(params: _TestStageDef_Outs) -> None:
        params.outs.result = {"value": 42}

    out_spec = outputs.Out(str(output_file))

    stage_info: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:test": "abc123"},
        "deps": [],
        "signature": None,
        "outs": [out_spec],
        "params": _TestStageDef_Outs(),
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "ran"
    assert output_file.exists()
    assert "42" in output_file.read_text()


def test_stage_def_missing_output_returns_failed(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """StageDef should fail if output was declared but never assigned."""
    output_file = tmp_path / "output.json"

    def stage_func(params: _TestStageDef_Outs) -> None:
        pass  # Never assign params.outs.result

    out_spec = outputs.Out(str(output_file))

    stage_info: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:test": "abc123"},
        "deps": [],
        "signature": None,
        "outs": [out_spec],
        "params": _TestStageDef_Outs(),
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "failed"
    assert "never assigned" in result["reason"]


def test_stage_def_load_failure_returns_failed(
    worker_env: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """StageDef should fail if deps cannot be loaded."""

    def stage_func(params: _TestStageDef_MissingDeps) -> None:
        pass

    stage_info: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:test": "abc123"},
        "deps": [str(tmp_path / "missing.csv")],  # File doesn't exist
        "signature": None,
        "outs": [],
        "params": _TestStageDef_MissingDeps(),
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    # Should fail before function runs (missing deps check)
    assert result["status"] == "failed"


def test_plain_params_no_auto_load_save(worker_env: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Plain Pydantic params should still work without auto-load/save."""
    output_file = tmp_path / "output.txt"

    def stage_func(params: _PlainParams) -> None:
        output_file.write_text(f"threshold: {params.threshold}")

    out_spec = outputs.Out(str(output_file))

    stage_info: WorkerStageInfo = {
        "func": stage_func,
        "fingerprint": {"self:test": "abc123"},
        "deps": [],
        "signature": None,
        "outs": [out_spec],
        "params": _PlainParams(),
        "variant": None,
        "overrides": {},
        "cwd": None,
        "checkout_modes": ["hardlink", "symlink", "copy"],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "no_cache": False,
    }

    result = executor.execute_stage(
        "test_stage",
        stage_info,
        worker_env,
        mp.Manager().Queue(),  # pyright: ignore[reportArgumentType]
    )

    assert result["status"] == "ran"
    assert output_file.exists()
    assert "threshold: 0.5" in output_file.read_text()
